/-
  Moyasser AI — Formally Verified Egyptian Tax Facilitation Law Kernel (Lean 4)
  Covers:
    - Law No. 6 of 2025 (Simplified SME Regime <= 20,000,000 EGP, 5 progressive turnover brackets)
    - Law No. 5 of 2025 (Taxpayer regularization & 100% delay-penalty waiver upon settlement)
    - Law No. 7 of 2025 (Delay-penalty statutory ceiling <= 100% of principal tax)
    - Second Tax Facilitation Package (2.5% Real Estate Disposal Tax + 1st-degree relative exemption)
  Zero floating-point drift (exact integer basis points: 100 bps = 1%, 10000 bps = 100%).
  Zero `sorry` placeholders: every theorem is machine-checked by the Lean 4 kernel.
-/

namespace MoyasserTaxLaw

/-- Statutory turnover ceiling under Law No. 6 of 2025 (20,000,000 EGP). -/
def law6Ceiling : Nat := 20000000

/-- Statutory one-time 20% grace buffer ceiling under Law No. 6 of 2025 (24,000,000 EGP). -/
def law6GraceCeiling : Nat := 24000000

/-- Law No. 6 of 2025 progressive turnover tax rate in basis points (100 bps = 1%). -/
def law6RateBps (revenue : Nat) : Option Nat :=
  if revenue < 500000 then some 40
  else if revenue < 2000000 then some 50
  else if revenue < 3000000 then some 75
  else if revenue < 10000000 then some 100
  else if revenue < 20000000 then some 150
  else none

/-- Exact simplified annual tax (EGP integer floor) under Law No. 6 of 2025. -/
def law6Tax (revenue : Nat) : Option Nat :=
  match law6RateBps revenue with
  | some bps => some ((revenue * bps) / 10000)
  | none => none

/-- Law No. 7 of 2025 statutory cap: delay penalty cannot exceed 100% of original tax principal (when principal > 0). -/
def law7CappedPenalty (principal rawPenalty : Nat) : Nat :=
  if principal = 0 then rawPenalty else Nat.min principal rawPenalty

/-- Law No. 5 of 2025 regularization: 100% waiver of delay penalty when compliant settlement applies. -/
def law5SettledPenalty (principal rawPenalty : Nat) (eligibleForWaiver : Bool) : Nat :=
  if eligibleForWaiver then 0 else law7CappedPenalty principal rawPenalty

/-- Package #2: Real Estate Disposal Tax (2.5% = 250 bps), strictly 0 EGP for 1st-degree relatives. -/
def realEstateDisposalTax (saleValue : Nat) (isFirstDegreeRelative : Bool) : Nat :=
  if isFirstDegreeRelative then 0 else (saleValue * 250) / 10000

/-- Optimal annual income/turnover tax between eligible Simplified Law 6 and Standard Regime. -/
def chosenOptimalTax (revenue stdTax : Nat) : Nat :=
  match law6Tax revenue with
  | some simTax => Nat.min simTax stdTax
  | none => stdTax

/-- Guaranteed non-negative savings of optimal regime vs. the alternative regime. -/
def optimalSavingsVsStandard (revenue stdTax : Nat) : Nat :=
  stdTax - chosenOptimalTax revenue stdTax

/-- Regime code chosen by the formally verified kernel. -/
def chosenRegimeCode (revenue stdTax : Nat) : String :=
  match law6Tax revenue with
  | some simTax => if simTax ≤ stdTax then "SIMPLIFIED_LAW6" else "STANDARD"
  | none => "STANDARD"

/-- Law No. 6 of 2025 turnover rate with optional one-time 20% grace buffer (<= 24M EGP at 150 bps). -/
def law6RateWithGraceBps (revenue : Nat) (graceYearEligible : Bool) : Option Nat :=
  if revenue < 20000000 then law6RateBps revenue
  else if graceYearEligible && revenue ≤ law6GraceCeiling then some 150
  else none

/-- Law No. 149 of 2026 (Second Package): VAT on medical equipment reduced from 14% (1400 bps) to 5% (500 bps). -/
def medicalEquipmentVatTax (value : Nat) (isDialysisInputExempt : Bool) : Nat :=
  if isDialysisInputExempt then 0 else (value * 500) / 10000

/-! ### Formal Machine-Checked Theorems (0 `sorry`) -/

/-- Theorem 1: Law 6 eligibility holds if and only if annual revenue is strictly below 20,000,000 EGP. -/
theorem law6_eligibility_iff (r : Nat) :
    (law6RateBps r).isSome = true ↔ r < law6Ceiling := by
  unfold law6RateBps law6Ceiling
  by_cases h1 : r < 500000
  · simp [h1]; omega
  · by_cases h2 : r < 2000000
    · simp [h1, h2]; omega
    · by_cases h3 : r < 3000000
      · simp [h1, h2, h3]; omega
      · by_cases h4 : r < 10000000
        · simp [h1, h2, h3, h4]; omega
        · by_cases h5 : r < 20000000
          · simp [h1, h2, h3, h4, h5]
          · simp [h1, h2, h3, h4, h5]

/-- Theorem 2: Every Law 6 bracket rate lies strictly within [0.4% (40 bps), 1.5% (150 bps)]. -/
theorem law6_rate_bounds (r b : Nat) (h : law6RateBps r = some b) :
    40 ≤ b ∧ b ≤ 150 := by
  unfold law6RateBps at h
  split at h
  · injection h with hb; omega
  · split at h
    · injection h with hb; omega
    · split at h
      · injection h with hb; omega
      · split at h
        · injection h with hb; omega
        · split at h
          · injection h with hb; omega
          · contradiction

/-- Theorem 3: Law 6 bracket rates are monotonically non-decreasing with annual revenue. -/
theorem law6_rate_monotone (r1 r2 b1 b2 : Nat)
    (hle : r1 ≤ r2)
    (h1 : law6RateBps r1 = some b1)
    (h2 : law6RateBps r2 = some b2) :
    b1 ≤ b2 := by
  have hb2_min : 40 ≤ b2 := (law6_rate_bounds r2 b2 h2).1
  unfold law6RateBps at h1 h2
  split at h1
  · injection h1 with hb1; omega
  · split at h1
    · injection h1 with hb1
      split at h2
      · omega
      · split at h2
        · injection h2 with hb2; omega
        · split at h2
          · injection h2 with hb2; omega
          · split at h2
            · injection h2 with hb2; omega
            · split at h2
              · injection h2 with hb2; omega
              · contradiction
    · split at h1
      · injection h1 with hb1
        split at h2
        · omega
        · split at h2
          · omega
          · split at h2
            · injection h2 with hb2; omega
            · split at h2
              · injection h2 with hb2; omega
              · split at h2
                · injection h2 with hb2; omega
                · contradiction
      · split at h1
        · injection h1 with hb1
          split at h2
          · omega
          · split at h2
            · omega
            · split at h2
              · omega
              · split at h2
                · injection h2 with hb2; omega
                · split at h2
                  · injection h2 with hb2; omega
                  · contradiction
        · split at h1
          · injection h1 with hb1
            split at h2
            · omega
            · split at h2
              · omega
              · split at h2
                · omega
                · split at h2
                  · omega
                  · split at h2
                    · injection h2 with hb2; omega
                    · contradiction
          · contradiction

/-- Theorem 4 (Law 7 of 2025): Statutory capped delay penalty never exceeds 100% of original tax principal. -/
theorem law7_penalty_cap_bound (principal rawPenalty : Nat) (hp : principal > 0) :
    law7CappedPenalty principal rawPenalty ≤ principal := by
  unfold law7CappedPenalty
  split
  · omega
  · exact Nat.min_le_left principal rawPenalty

/-- Theorem 5 (Law 5 of 2025): Regularization penalty is always <= Law 7 statutory capped penalty. -/
theorem law5_waiver_reduces_burden (principal rawPenalty : Nat) (elig : Bool) :
    law5SettledPenalty principal rawPenalty elig ≤ law7CappedPenalty principal rawPenalty := by
  unfold law5SettledPenalty
  cases elig <;> simp

/-- Theorem 6 (Second Package): First-degree relative real-estate disposal is strictly tax-exempt (0 EGP). -/
theorem real_estate_first_degree_exempt (saleValue : Nat) :
    realEstateDisposalTax saleValue true = 0 := by
  rfl

/-- Theorem 7 (Optimality Soundness): Chosen tax never exceeds Standard regime tax. -/
theorem optimal_never_exceeds_standard (revenue stdTax : Nat) :
    chosenOptimalTax revenue stdTax ≤ stdTax := by
  unfold chosenOptimalTax
  cases law6Tax revenue with
  | none => exact Nat.le_refl stdTax
  | some simTax => exact Nat.min_le_right simTax stdTax

/-- Theorem 8 (Savings Conservation): Optimal tax + guaranteed savings equals Standard tax exactly. -/
theorem savings_conservation (revenue stdTax : Nat) :
    chosenOptimalTax revenue stdTax + optimalSavingsVsStandard revenue stdTax = stdTax := by
  unfold optimalSavingsVsStandard
  have h := optimal_never_exceeds_standard revenue stdTax
  omega

/-- Theorem 9 (Law 6 of 2025 Grace Buffer): One-time 20% grace rule extends eligibility strictly up to 24,000,000 EGP. -/
theorem law6_grace_buffer_ceiling (r : Nat) (h : (law6RateWithGraceBps r true).isSome = true) :
    r ≤ law6GraceCeiling := by
  unfold law6GraceCeiling
  unfold law6RateWithGraceBps law6GraceCeiling at h
  by_cases h20 : r < 20000000
  · omega
  · simp [h20] at h
    exact h

/-- Theorem 10 (Law 149 of 2026 Medical VAT Relief): Reduced 5% medical VAT (or 0% dialysis exemption) never exceeds standard 14% VAT. -/
theorem law149_medical_vat_reduction (val : Nat) (dialysis : Bool) :
    medicalEquipmentVatTax val dialysis ≤ (val * 1400) / 10000 := by
  unfold medicalEquipmentVatTax
  cases dialysis <;> simp <;> omega

end MoyasserTaxLaw
