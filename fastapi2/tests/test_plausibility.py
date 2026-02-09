import unittest
import sys
import os

sys.path.append(os.getcwd())

from app.core.validation.biomarker_plausibility import BiomarkerPlausibilityValidator, BiomarkerSet, Biomarker, PhysiologicalSystem, ViolationType

class TestPlausibility(unittest.TestCase):
    def setUp(self):
        self.validator = BiomarkerPlausibilityValidator()

    def test_hrv_tachycardia_logic(self):
        # Case 1: High HR, Low HRV (Normal for Tachycardia) - Should NOT flag
        bm_set = BiomarkerSet(
            system=PhysiologicalSystem.CARDIOVASCULAR,
            biomarkers=[
                Biomarker(name="heart_rate", value=150.0, unit="bpm"),
                Biomarker(name="hrv_rmssd", value=20.0, unit="ms")
            ]
        )
        result = self.validator.validate(bm_set)
        contradictions = [v for v in result.violations if v.violation_type == ViolationType.INTERNAL_CONTRADICTION]
        self.assertEqual(len(contradictions), 0, "Should not flag normal tachycardia")

        # Case 2: High HR, High HRV (Suspicious) - Should FLAG
        bm_set_bad = BiomarkerSet(
            system=PhysiologicalSystem.CARDIOVASCULAR,
            biomarkers=[
                Biomarker(name="heart_rate", value=150.0, unit="bpm"),
                Biomarker(name="hrv_rmssd", value=100.0, unit="ms")
            ]
        )
        result_bad = self.validator.validate(bm_set_bad)
        contradictions = [v for v in result_bad.violations if "High HR" in v.message]
        self.assertEqual(len(contradictions), 1, "Should flag High HR + High HRV")

    def test_sudden_jump_logic(self):
        # Previous set
        prev_set = BiomarkerSet(
            system=PhysiologicalSystem.CARDIOVASCULAR,
            biomarkers=[Biomarker(name="heart_rate", value=80.0, unit="bpm")],
            extraction_time_ms=1000
        )
        
        # Current set: 0.1s later, jump by 5 bpm (Rate = 50 bpm/s -> Should flag as >30 bpm/s)
        # 5 bpm in 0.1s = 50 bpm/s
        curr_set = BiomarkerSet(
            system=PhysiologicalSystem.CARDIOVASCULAR,
            biomarkers=[Biomarker(name="heart_rate", value=85.0, unit="bpm")],
            extraction_time_ms=1100 # 100ms later
        )
        
        # Result
        result = self.validator.validate(curr_set, previous_set=prev_set)
        jump_violations = [v for v in result.violations if v.violation_type == ViolationType.SUDDEN_JUMP]
        
        self.assertEqual(len(jump_violations), 1, "Should flag sudden jump")
        self.assertTrue("changed too quickly" in jump_violations[0].message)

if __name__ == '__main__':
    unittest.main()
