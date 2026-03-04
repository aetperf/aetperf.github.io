---
title: "My First Lean 4 Proof: the Irrationality of √2"
layout: post
comments: true
author: François Pacull
categories: [formal verification, Lean]
tags:
- Lean
- Lean 4
- formal verification
- Mathlib
- proof assistant
- theorem proving
- mathematics
- irrationality
- Claude Code
---

<figure style="text-align: center;">
  <img width="400" src="https://lean-lang.org/static/png/lean-logo-official-TM-white-2400x900.png" alt="Lean logo" style="display: block; margin: 0 auto;">
</figure>

Formal verification uses computers to automatically check whether proofs are correct. [Lean](https://lean-lang.org/) is a formal proof assistant where we write proofs as code, and it tells you if your reasoning is correct. I picked a classic theorem: *$\sqrt{2}$ is irrational*, and used [Claude Code](https://docs.anthropic.com/en/docs/claude-code) to help prove it in Lean, as a learning exercise.

## Getting started on Linux

I installed `elan` (Lean's version manager) with `curl`:

```bash
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
```

This sets up `lean` and `lake` (the build tool):

```bash
$ elan --version
elan 4.1.2 (58e8d545e 2025-05-26)

$ lean --version
Lean (version 4.27.0, x86_64-unknown-linux-gnu, commit db93fe1608548, Release)

$ lake --version
Lake version 5.0.0-src+db93fe1 (Lean version 4.27.0)
```

```bash
mkdir my_lean_project && cd my_lean_project
lake init my_lean_project math  # Create project with Mathlib as dependency
lake exe cache get  # Download Mathlib's pre-built cache
```

`lake exe cache get` downloads pre-built binaries so I didn't have to compile Mathlib from source. That alone saved a lot of time. Mathlib is around 2 million lines of code. From [Wikipedia](https://en.wikipedia.org/wiki/Lean_(proof_assistant)):

> In 2017, a community-maintained project to develop a Lean library *mathlib* began, with the goal to digitize as much of pure mathematics as possible in one large cohesive library, up to research level mathematics. As of May 2025, mathlib had formalized over 210,000 theorems and 100,000 definitions in Lean.

Any text editor works for `.lean` files. A typical workflow in the terminal looks like:

- Running `lake build` to see errors
- Using `sorry` to see what sub-goals remain
- Sprinkling `#check` and `#print` commands temporarily

`lake init` automatically converts the project name to PascalCase for the source directory. So `my_lean_project` creates a `MyLeanProject/` folder:

```
my_lean_project/              # top-level directory
├── lakefile.toml             # build config (Mathlib dependency declared here)
├── lake-manifest.json        # auto-generated, locks dependency versions
├── lean-toolchain            # pinned Lean version
├── MyLeanProject.lean        # root import file (lists modules)
└── MyLeanProject/            # in PascalCase
    └── Sqrt2.lean            # source files go here
```

## The full proof

Mathlib already ships a one-liner for this: [`irrational_sqrt_two`](https://leanprover-community.github.io/mathlib4_docs/Mathlib/NumberTheory/Real/Irrational.html#irrational_sqrt_two). The point here is to rebuild the classic argument showing both numerator and denominator must be even from scratch, tactic by tactic, to see how Lean works.

The proof follows the classic coprimality contradiction:

1. Assume $\sqrt{2} = \frac{n}{m}$ with $\gcd(m, n) = 1$
2. Squaring gives $2m^2 = n^2$, so $2 \mid n$
3. Substitute $n = 2k$, which gives $m^2 = 2k^2$, so $2 \mid m$
4. Both $m$ and $n$ divisible by 2 contradicts $\gcd(m, n) = 1$

Here's the full proof from `Sqrt2.lean`:

```text
import Mathlib.Data.Nat.GCD.Basic
import Mathlib.Tactic

-- Helper: if n² is even, then n is even
lemma even_of_even_sq {n : ℕ} (h : 2 ∣ n ^ 2) : 2 ∣ n := by
  exact (Nat.Prime.dvd_of_dvd_pow Nat.prime_two h)

-- Main theorem: no coprime m, n satisfy 2 * m² = n²
theorem sqrt2_irrat : ¬ ∃ m n : ℕ, 2 * m ^ 2 = n ^ 2 ∧ Nat.Coprime m n := by
  rintro ⟨m, n, hmn, hcop⟩
  have h2n : 2 ∣ n ^ 2 := ⟨m ^ 2, by linarith⟩
  have hn : 2 ∣ n := even_of_even_sq h2n
  obtain ⟨k, rfl⟩ := hn
  have h2m : 2 ∣ m ^ 2 := by
    have : 2 * m ^ 2 = 4 * k ^ 2 := by ring_nf at hmn ⊢; linarith
    exact ⟨k ^ 2, by linarith⟩
  have hm : 2 ∣ m := even_of_even_sq h2m
  exact Nat.not_coprime_of_dvd_of_dvd (by norm_num) hm ⟨k, rfl⟩ hcop

#print axioms sqrt2_irrat
```

This was tested with Lean 4.27.0 and Mathlib v4.27.0. To verify it, run `lake build` from the project root:

```
$ lake build
info: MyLeanProject/Sqrt2.lean:24:0: 'sqrt2_irrat' depends on axioms: [propext, Classical.choice, Quot.sound]
Build completed successfully.
```

Zero errors, zero `sorry`s: the proof is machine-checked.

The `#print axioms` line at the end asks Lean to list the foundational axioms the proof relies on. The three it reports, `propext` (propositional extensionality), `Classical.choice` (the axiom of choice), and `Quot.sound` (quotient soundness), are Lean's standard axioms. Every non-trivial Mathlib proof depends on them.

Let's walk through the code.

## Step by step

### Imports

The two `import` lines are the same as in the full proof above. `import` loads Mathlib modules. The first brings in facts about GCD and coprimality of natural numbers. The second gives us general-purpose tactics like `linarith`, `ring_nf`, and `norm_num`.

In Lean, we can write proofs in two ways. In term mode, we construct the proof directly as an expression, like writing a program that returns the right type. In tactic mode, we start from a goal (the thing we want to prove) and apply step-by-step commands that transform it until nothing is left to prove. Each command is a tactic. Something like `exact` just hands Lean the answer directly. Others do real work: `linarith` solves goals by linear arithmetic, `rintro` breaks apart hypotheses into pieces.

We enter tactic mode by writing `by`, and from there it's one tactic per line, each one chipping away at the goal. This proof uses tactic mode throughout.

### The helper lemma: if $n^2$ is even, then $n$ is even

```text
lemma even_of_even_sq {n : ℕ} (h : 2 ∣ n ^ 2) : 2 ∣ n := by
  exact (Nat.Prime.dvd_of_dvd_pow Nat.prime_two h)
```

`lemma` declares a reusable fact. The curly braces `{n : ℕ}` make `n` an implicit argument; Lean infers it from context. The parentheses `(h : 2 ∣ n ^ 2)` are an explicit hypothesis: a proof that $2 \mid n^2$. After the colon, `2 ∣ n` is what we want to prove.

`exact` says "this is the proof term, verbatim." We hand it a Mathlib theorem: if a prime divides a power, it divides the base. Since 2 is prime (`Nat.prime_two`), we get $2 \mid n$.

Mathlib lemma names can change between versions. If `Nat.Prime.dvd_of_dvd_pow` doesn't resolve in yours, replace the proof line with `exact?`. It searches Mathlib for a term that matches the current goal. In VS Code with the Lean extension, suggestions appear in the infoview panel in real time. It also works via `lake build`, where the suggestion shows up as an info message in the build output, but this is slower. Either way, `exact?` can take a while since it scans a large portion of Mathlib. A related tactic, `apply?`, searches for lemmas that can be applied to the goal but may leave sub-goals to fill in, rather than solving it completely.

### The main theorem

```text
theorem sqrt2_irrat : ¬ ∃ m n : ℕ, 2 * m ^ 2 = n ^ 2 ∧ Nat.Coprime m n := by
```

`theorem` is like `lemma` but signals a main result. `¬` is negation, `∃` is "there exists," `∧` is "and." So this reads: there do not exist natural numbers $m, n$ such that $2m^2 = n^2$ and $\gcd(m, n) = 1$.

### Step 1 — Assume the opposite

```text
  rintro ⟨m, n, hmn, hcop⟩
```

`rintro` peels apart the structure of the goal. Since we're proving a negation (`¬ ∃ ...`), it assumes the existential and destructs it in one go. The angle brackets `⟨...⟩` are anonymous constructor syntax. The four names match the structure of `∃ m n : ℕ, 2 * m ^ 2 = n ^ 2 ∧ Nat.Coprime m n` left to right:

- `m` and `n` bind the two existential witnesses
- `hmn` names the proof of `2 * m ^ 2 = n ^ 2` (left of `∧`)
- `hcop` names the proof of `Nat.Coprime m n` (right of `∧`)

These are not keywords — any names would work. The `h` prefix is just a Lean convention for hypotheses.

### Step 2 — Show $2 \mid n$

```text
  have h2n : 2 ∣ n ^ 2 := ⟨m ^ 2, by linarith⟩
  have hn : 2 ∣ n := even_of_even_sq h2n
```

`have` introduces a new fact into the proof context. In Lean, `2 ∣ n ^ 2` is by definition `∃ k, n ^ 2 = 2 * k`, so the anonymous constructor `⟨m ^ 2, by linarith⟩` provides the witness directly: we claim $k = m^2$. `linarith` verifies that the witness satisfies the divisibility condition, given `hmn`.

Then we apply our helper lemma to go from $2 \mid n^2$ to $2 \mid n$.

### Step 3 — Substitute $n = 2k$

```text
  obtain ⟨k, rfl⟩ := hn
```

`obtain` destructs the proof that $2 \mid n$ into a witness: some $k$ and a proof that $n = 2k$. The `rfl` pattern is the trick: instead of keeping the equation `n = 2k` as a hypothesis, it immediately substitutes $2k$ for $n$ everywhere.

### Step 4 — Show $2 \mid m$

```text
  have h2m : 2 ∣ m ^ 2 := by
    have : 2 * m ^ 2 = 4 * k ^ 2 := by ring_nf at hmn ⊢; linarith
    exact ⟨k ^ 2, by linarith⟩
  have hm : 2 ∣ m := even_of_even_sq h2m
```

Now that $n = 2k$, the equation `hmn` became $2m^2 = (2k)^2$. `ring_nf` normalizes ring expressions; here it expands $(2k)^2$ into $4k^2$. The `at hmn ⊢` syntax means "apply this to both the hypothesis `hmn` and the current goal (`⊢`)." Then `linarith` finishes the arithmetic. We need to normalize both because, after substituting $n = 2k$, the hypothesis and goal are in different algebraic forms that `linarith` can't directly compare. `linarith` does not expand nonlinear expressions like $(2k)^2$, so `ring_nf` must rewrite it to $4 \; k^2$ first.


### Step 5 — Contradiction

```text
  exact Nat.not_coprime_of_dvd_of_dvd (by norm_num) hm ⟨k, rfl⟩ hcop
```

We now have $2 \mid m$ and $2 \mid n$, but we assumed $\gcd(m, n) = 1$. The Mathlib lemma `Nat.not_coprime_of_dvd_of_dvd` says: if some $d > 1$ divides both, they can't be coprime. `norm_num` verifies that $2 > 1$, we supply the divisibility proofs, and Lean closes the goal. The `⟨k, rfl⟩` here re-packages the fact that $n = 2k$, which became definitional after the earlier `obtain` substitution. $\square$

One subtlety: the proof never uses `by_contra`. The goal is already a negation (`¬ ∃ ...`), which in Lean unfolds to `(∃ ...) → False`. So `rintro` just introduces the antecedent and we derive `False` directly. This is proving a negation, not proof by contradiction.

### What happens when it's wrong

What if we skip a step? Here's the same proof with the "show $2 \mid m$" block deleted; we go straight from $n = 2k$ to the contradiction:

```text
  obtain ⟨k, rfl⟩ := hn
  -- Step 4 (show 2 ∣ m) is missing here
  exact Nat.not_coprime_of_dvd_of_dvd (by norm_num) hm ⟨k, rfl⟩ hcop
```

Lean refuses to build it:

```
$ lake build MyLeanProject.Sqrt2_wrong
error: MyLeanProject/Sqrt2_wrong.lean:13:52: Unknown identifier `hm`
```

We never proved $2 \mid m$, so `hm` doesn't exist. Lean catches the gap immediately. Every step must be justified; you cannot skip from $2 \mid n$ to the contradiction without proving $2 \mid m$ first.

## Conclusion

This was a small proof, but a satisfying one to get right. Every step mirrors the classical pen-and-paper argument. Lean isn't inventing a new proof; it is forcing us to justify each algebraic move explicitly.

The combination of Lean and AI is moving fast. AI models are getting better at generating proofs, and Lean provides an automatic way to verify them.

## Appendix: Lean tactics reference

### Types and terms

Everything in Lean has a type. To prove a proposition `P`, you construct a value whose type is that proposition `P`. If Lean accepts the value, the proposition is proved.

```text
-- A simple theorem: for all natural numbers n, 0 + n = n
theorem zero_add (n : Nat) : 0 + n = n := by
  simp
```

This is referred to as the [Curry-Howard correspondence](https://en.wikipedia.org/wiki/Curry%E2%80%93Howard_correspondence).

The `rintro ⟨m, n, hmn, hcop⟩` line in our proof above is really just pattern matching on a proof object. Since `¬ A` means `A → False`, the theorem is actually a function that takes a value of type `∃ m n, 2 * m^2 = n^2 ∧ Nat.Coprime m n` and returns a contradiction. Here is a hybrid version: the overall structure is term-mode (the proof is a function starting with `fun ⟨m, n, hmn, hcop⟩ =>`), but `by` blocks still handle the arithmetic via tactics:

```
import Mathlib.Data.Nat.GCD.Basic
import Mathlib.Tactic

lemma even_of_even_sq {n : ℕ} (h : 2 ∣ n ^ 2) : 2 ∣ n :=
  Nat.Prime.dvd_of_dvd_pow Nat.prime_two h

theorem sqrt2_irrat :
  ¬ ∃ m n : ℕ, 2 * m ^ 2 = n ^ 2 ∧ Nat.Coprime m n :=
fun ⟨m, n, hmn, hcop⟩ =>

  -- 2 ∣ n²
  let h2n : 2 ∣ n ^ 2 :=
    ⟨m ^ 2,
      show n ^ 2 = 2 * m ^ 2 from
        hmn.symm⟩

  -- 2 ∣ n
  let hn : 2 ∣ n :=
    even_of_even_sq h2n

  -- destruct divisibility of n
  match hn with
  | ⟨k, hk⟩ =>

    -- rewrite n = 2k inside equation
    let hmn' : 2 * m ^ 2 = (2 * k) ^ 2 :=
      hk ▸ hmn

    -- normalize RHS manually via arithmetic reasoning
    let h2m : 2 ∣ m ^ 2 :=
      ⟨k ^ 2,
        by
          have : 2 * m ^ 2 = 4 * k ^ 2 := by
            have := hmn'
            ring_nf at this
            exact this
          linarith⟩

    -- 2 ∣ m
    let hm : 2 ∣ m :=
      even_of_even_sq h2m

    -- contradiction with coprimality
    Nat.not_coprime_of_dvd_of_dvd
      (by norm_num)
      hm
      ⟨k, hk⟩
      hcop
```

Each by block enters tactic mode and uses tactics like `ring_nf`, `norm_num`, and `linarith`, which come from `Mathlib.Tactic`. So even though the overall structure is term-mode, the arithmetic heavy-lifting is still delegated to tactics. A fully pure term-mode proof would need to construct those arithmetic witnesses by hand, which would be significantly more verbose.

### Tactics

Proofs in Lean are ultimately terms. Tactics are a convenient way to build those terms interactively by transforming the proof goal step by step:

| Tactic | What it does |
|--------|-------------|
| `intro h` | Introduce a hypothesis |
| `apply f` | Apply a lemma to the goal |
| `exact e` | Provide the exact proof term |
| `have h : T := e` | Introduce a new fact into the context |
| `obtain ⟨a, b⟩ := h` | Destruct a hypothesis into components |
| `rintro ⟨a, b⟩` | Introduce and destruct in one step |
| `simp` | Simplify using known lemmas |
| `ring` | Prove equalities in commutative rings |
| `ring_nf` | Normalize ring expressions without closing the goal |
| `omega` | Decide some Presburger arithmetic goals |
| `linarith` | Linear arithmetic reasoning |
| `norm_num` | Evaluate numeric expressions |
| `contradiction` | Close a goal from contradictory hypotheses |
| `rcases h with ⟨a, b⟩` | Destructure an existential or conjunction |
| `by_contra h` | Proof by contradiction |
| `exact?` | Search Mathlib for a matching proof term |
| `apply?` | Search Mathlib for an applicable lemma |

### The `sorry` placeholder

`sorry` lets us skip a proof step temporarily. Lean will warn us, but the file still compiles. Use it to explore what sub-goals remain:

```text
theorem my_theorem : 1 + 1 = 2 := by
  sorry  -- Lean warns but compiles; fill in later
```
