---
title: German-style strings in Apache Arrow
layout: post
comments: true
author: François Pacull
date: 2026-05-29
categories: [Arrow]
tags:
- Apache Arrow
- StringView
- BinaryView
- Columnar Format
- German-style strings
- Umbra
- PyArrow
image: /img/2026-05-29_01/fig_03.png
---

Version 1.4 of the [Apache Arrow](https://arrow.apache.org/) Columnar Format added a new way to store variable-length values, the [Variable-size Binary View Layout](https://arrow.apache.org/docs/format/Columnar.html#variable-size-binary-view-layout). It comes in two types: `BinaryView` for raw bytes and `StringView` for UTF-8 text. They work like the older `Binary` and `String` types and use the same memory layout, except that `StringView` requires its bytes to be valid UTF-8. The name *view* comes from how each value is stored: a small fixed-size record of 16 bytes. Short values sit inside the record; longer ones are kept in a separate buffer that the record points to. Operations move these small records around, while the string bytes themselves never move, and several records can point at the same bytes. The format is at version 1.5 today.

The layout is Arrow's take on the *German-style string* described in the [Umbra paper](https://db.in.tum.de/~freitag/papers/p29-neumann-cidr20.pdf) by Neumann and Freitag [2], and already used by [Velox](https://velox-lib.io/) and [DuckDB](https://duckdb.org/). For long strings, the record also keeps a copy of the first 4 bytes of the value, which lets many comparisons finish without reading the full string. I first read about this layout in Topol's *In-Memory Analytics with Apache Arrow* [1] (figures 1.10 and 1.11), a book I have enjoyed a lot.

## The classic variable-length layout

Arrow's older `String`/`Binary` type stores its values in three buffers: a validity bitmap (which values are null), an offsets buffer of `n + 1` integers, and one values buffer holding all the bytes back to back. Value `i` is the slice of the values buffer between `offsets[i]` and `offsets[i + 1]`. The offsets only ever grow, so the strings are stored in order.

<p align="center">
  <img src="/img/2026-05-29_01/fig_01.png" alt="Classic Arrow String layout" width="700" />
</p>

*Reading any value by its index is instant, but each string sits at one fixed spot in one buffer; slicing can just adjust the offsets, while sorting has to rewrite the values buffer too.*

## The view header

In a `StringView` array, there is no offsets buffer. Each value is described by a fixed-size record of 16 bytes, called a *view header*. The first 4 bytes always hold the length. If the string is 12 bytes or shorter, the remaining 12 bytes hold it directly. If it is longer, the string is kept elsewhere and the remaining 12 bytes hold three things: a copy of its first 4 bytes, the number of the value buffer it lives in, and where it starts inside that buffer. The 16-byte size is not arbitrary either: it is the width of a single 128-bit CPU register, so a whole header can be loaded and compared in one instruction. The 12-byte inline limit is just what is left after the 4-byte length.

<p align="center">
  <img src="/img/2026-05-29_01/fig_02.png" alt="16-byte view header" width="700" />
</p>

*Short strings (12 bytes or fewer) sit inside the header; long strings keep their first 4 bytes plus a (buffer, offset) pair pointing to the rest.*

The length, buffer index, and offset are signed 32-bit integers, and the prefix is 4 bytes copied from the start of the value. Arrow uses signed integers for these indices because some languages, notably Java, have no unsigned integer types. The prefix must always match the real start of the string. The shortcut described below depends on this: if a long view's prefix and its string disagree, the array is broken.

## Multiple value buffers

Unlike every other Arrow layout, the view layout allows a *variable* number of buffers per array. After the validity bitmap and the views buffer come zero or more value buffers, and each long view says which one it points to by its number. The buffers can be any size, and the same bytes can be shared by several views, in any order.

<p align="center">
  <img src="/img/2026-05-29_01/fig_03.png" alt="Full StringView array" width="900" />
</p>

*The two long views `[1]` and `[4]` point into separate value buffers and share the same 4-byte prefix `Ich `.*

This layout has three practical benefits. First, the views buffer can be reordered, sliced, or filtered without touching any string bytes; only the 16-byte headers move. Sorting and filtering rewrite the views buffer but leave the value buffers as they are. Joining two arrays together is just as cheap: keep both sets of value buffers, put the two views buffers one after the other, and renumber the buffers referenced by the second array's long views.

Second, the 4-byte prefix lets you compare two strings without reading them in full. If the prefixes differ, the strings differ, and there is no need to go fetch the rest. This makes it a quick filter: when scanning a column for a given value, any row whose prefix does not match is discarded on the spot, without a jump to the value buffer. The other direction does not hold: equal prefixes do not mean equal strings. `"Ich liebe dich"` and `"Ich liebe Bier"` share the prefix `Ich `, so the prefix check passes and the full strings still have to be read to tell them apart. The prefix is a one-way shortcut: helpful when it rules a comparison out, no help when it does not.

Third, when most strings in an array are short (12 bytes or fewer), every value sits inside its own header and there is no need to jump to a value buffer at all. The views buffer is read straight through; the jump only happens for the few long strings.

## A short PyArrow example

Constructing a `StringView` array from Python requires [pyarrow](https://arrow.apache.org/docs/python/) v14 or newer. pyarrow's version number tracks the Arrow C++ implementation it is built on, not the Columnar Format: the `24.0.0` below is the library release, while the format it implements is at 1.5. The two are versioned independently.

```python
import pyarrow as pa

print("pyarrow version : ",pa.__version__)

arr = pa.array(
    ["Hallo!", "Ich liebe dich", "Wunderbar!", None, "Ich liebe Bier"],
    type=pa.string_view(),
)

print(arr.type)
print(arr)
print("nbytes:", arr.nbytes)
```

```
pyarrow version :  24.0.0
string_view
[
  "Hallo!",
  "Ich liebe dich",
  "Wunderbar!",
  null,
  "Ich liebe Bier"
]
nbytes: 109
```

The 109 bytes break down as 1 byte of validity bitmap (`0b00010111`), 80 bytes of views (5 × 16), and 28 bytes of value data, the latter holding the two long strings of 14 bytes each. Decoding the views buffer confirms the layout:

| Slot  | Length | Storage     | Header contents |
|-------|--------|-------------|-----------------|
| `[0]` | 6      | inline      | `"Hallo!"` + 6 trailing bytes |
| `[1]` | 14     | out-of-line | prefix `"Ich "`, buffer 0, offset 0 |
| `[2]` | 10     | inline      | `"Wunderbar!"` + 2 trailing bytes |
| `[3]` | 0      | null slot   | header bytes unspecified |
| `[4]` | 14     | out-of-line | prefix `"Ich "`, buffer 0, offset 14 |

The Arrow format does not say what goes in the unused bytes of short views or in a null slot's header; pyarrow fills them with zeros, but readers should not count on this. PyArrow has put the two long values into a single 28-byte value buffer: `"Ich liebe dich"` at offset 0, then `"Ich liebe Bier"` at offset 14. The format allows any number of value buffers, and the constructor used one here. The two long views have the same prefix `Ich ` and the same length 14. Only their last 10 bytes differ, which is exactly the case the prefix shortcut cannot settle on its own.

## Tradeoffs

The view layout is not always better than the classic `String` type. Reading a long string takes an extra step, from the header to the value buffer. And after many slices and joins, the value buffers can fill up with bytes no view points to any more; that space is only reclaimed by an explicit cleanup pass. The prefix shortcut only helps when the first 4 bytes actually differ between values; for a column of URLs or paths that share the same host, those 4 bytes are the same everywhere and do no useful work.

## Availability

The view type was added to the Arrow Columnar Format in v1.4 (the current version is v1.5) and works from Arrow C++ / pyarrow v14 onward. Readers older than v1.4 reject files and streams that contain view arrays, so if you are writing for an older consumer, fall back to the classic `String` / `Binary` types (the Variable-size Binary Layout). The [implementation status page](https://arrow.apache.org/docs/status.html) tracks support in the other language libraries.

## References

- [1] Topol, M. (2024). *In-Memory Analytics with Apache Arrow*, 2nd edition. Packt Publishing. The chapter "Getting Started with Apache Arrow" describes the variable-length binary view array layout (figures 1.10 and 1.11).
- [2] Neumann, T., and Freitag, M. (2020). [Umbra: A Disk-Based System with In-Memory Performance](https://db.in.tum.de/~freitag/papers/p29-neumann-cidr20.pdf). *10th Annual Conference on Innovative Data Systems Research (CIDR)*. Section 3.1 introduces the 16-byte string header later adopted by Arrow.
- [3] Apache Arrow project. [Variable-size Binary View Layout](https://arrow.apache.org/docs/format/Columnar.html#variable-size-binary-view-layout). Authoritative specification.
