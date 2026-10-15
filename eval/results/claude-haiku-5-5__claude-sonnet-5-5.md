Items: 234 {'good': 83, 'bad_key': 83, 'ambiguous': 68}  
Solver accuracy on good items: 0.961  
Reviewer, shown solver votes: {'accepted_good': 1.0, 'rejected_ambiguous': 0.971, 'corrections_proposed': 82, 'corrections_right': 82, 'corrections_rechecked': 82, 'corrections_confirmed': 82, 'key_anchoring': {'pairs': 83, 'called_ambiguous_only_with_wrong_key': 1, 'rate': 0.012}}  
Reviewer, blind: {'accepted_good': 1.0, 'rejected_ambiguous': 0.985, 'corrections_proposed': 82, 'corrections_right': 81, 'corrections_rechecked': 82, 'corrections_confirmed': 82, 'key_anchoring': {'pairs': 83, 'called_ambiguous_only_with_wrong_key': 2, 'rate': 0.024}}  
Reviewer, keyless: {'accepted_good': 1.0, 'rejected_ambiguous': 1.0, 'corrections_proposed': 82, 'corrections_right': 82, 'corrections_rechecked': 82, 'corrections_confirmed': 82, 'key_anchoring': {'pairs': 83, 'called_ambiguous_only_with_wrong_key': 1, 'rate': 0.012}}

| configuration | k | shipped | error rate (shipped) | good yield | bad keys caught | ambiguous caught | USD | USD per correct |
|---|---|---|---|---|---|---|---|---|
| review_only | – | 165 | 0.006 | 1.0 | 1.0 | 0.985 | 1.4227 | 0.00868 |
| review_only_keyless | – | 165 | 0.0 | 1.0 | 1.0 | 1.0 | 1.6954 | 0.01027 |
| gate_only | 1 | 145 | 0.434 | 0.988 | 0.964 | 0.118 | 0.0553 | 0.00067 |
| gate_only | 2 | 139 | 0.417 | 0.976 | 0.988 | 0.162 | 0.0553 | 0.00068 |
| gate_only | 3 | 135 | 0.415 | 0.952 | 0.988 | 0.191 | 0.0553 | 0.0007 |
| gate_only | 4 | 130 | 0.392 | 0.952 | 0.988 | 0.265 | 0.0553 | 0.0007 |
| gate_only | 5 | 125 | 0.376 | 0.94 | 0.988 | 0.324 | 0.0553 | 0.00071 |
| gate+review | 1 | 87 | 0.023 | 0.988 | 1.0 | 0.971 | 0.8408 | 0.00989 |
| gate+review | 2 | 84 | 0.024 | 0.976 | 1.0 | 0.971 | 0.795 | 0.00969 |
| gate+review | 3 | 82 | 0.024 | 0.952 | 1.0 | 0.971 | 0.7702 | 0.00963 |
| gate+review | 4 | 82 | 0.024 | 0.952 | 1.0 | 0.971 | 0.7369 | 0.00921 |
| gate+review | 5 | 81 | 0.025 | 0.94 | 1.0 | 0.971 | 0.7038 | 0.00891 |
| full | 1 | 166 | 0.012 | 0.988 | 1.0 | 0.971 | 1.4497 | 0.00884 |
| full | 2 | 165 | 0.012 | 0.976 | 1.0 | 0.971 | 1.4497 | 0.00889 |
| full | 3 | 163 | 0.012 | 0.952 | 1.0 | 0.971 | 1.4497 | 0.009 |
| full | 4 | 163 | 0.012 | 0.952 | 1.0 | 0.971 | 1.3916 | 0.00864 |
| full | 5 | 159 | 0.013 | 0.94 | 1.0 | 0.971 | 1.3053 | 0.00831 |
| gate+keyless | 1 | 85 | 0.0 | 0.988 | 1.0 | 1.0 | 1.0387 | 0.01222 |
| gate+keyless | 2 | 82 | 0.0 | 0.976 | 1.0 | 1.0 | 0.9976 | 0.01217 |
| gate+keyless | 3 | 80 | 0.0 | 0.952 | 1.0 | 1.0 | 0.9689 | 0.01211 |
| gate+keyless | 4 | 80 | 0.0 | 0.952 | 1.0 | 1.0 | 0.9301 | 0.01163 |
| gate+keyless | 5 | 79 | 0.0 | 0.94 | 1.0 | 1.0 | 0.895 | 0.01133 |
| full_keyless | 1 | 164 | 0.0 | 0.988 | 1.0 | 1.0 | 1.7506 | 0.01067 |
| full_keyless | 2 | 163 | 0.0 | 0.976 | 1.0 | 1.0 | 1.7506 | 0.01074 |
| full_keyless | 3 | 161 | 0.0 | 0.952 | 1.0 | 1.0 | 1.7506 | 0.01087 |
| full_keyless | 4 | 161 | 0.0 | 0.952 | 1.0 | 1.0 | 1.6831 | 0.01045 |
| full_keyless | 5 | 157 | 0.0 | 0.94 | 1.0 | 1.0 | 1.6004 | 0.01019 |
