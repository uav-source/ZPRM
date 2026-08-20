# Zero-perturbation mainline amendment v1.1 — PROPOSED, NOT ACTIVE

`status=PROPOSED_NOT_ACTIVE`  
`FORMAL_AUTHORITY=false`  
`FORMAL_REGISTRATION_AUTHORIZED=false`  
`FORMAL_ICP_UNLOCKED=false`  
`MEASUREMENT_FINAL_RESULT=false`

This document cannot unlock a backend, amend the active preregistration by implication, or turn any fixture into a formal result. The active `preregistration.yaml` and `analysis_protocol.md` remain byte-for-byte authoritative for their current scope.

## Proposed separation of tracks

The proposal would add an explicitly named `ZERO_PERTURBATION_MAINLINE` track as the proposed primary track after a valid W04 replacement and a complete re-freeze. That track would use each of 180 locked same-station QUERY snapshots against its locked MAP target with exact identity `T0`, once for Open3D and once for PCL: 360 locked trial records. Scene remains the highest independent unit; 180 snapshots are nested repeats, not 180 independent scenes.

Its proposed primary endpoints are translation displacement (norm and components), rotation displacement (angle and components), Weak-versus-Rich scene comparison, Open3D-versus-PCL agreement, correspondence reassociation/turnover, and systematic direction/component patterns.

The existing `CAPTURE_BASIN` track remains distinct. Under this unactivated proposal its proposed role would be optional supplementary, while its existing preregistration authority remains unchanged. It retains the preregistered translation magnitudes, weak/strong directions, signs, bracket refinement, and capture-radius endpoints. A perturbed capture-basin trial must use its exact locked `T0`; identity cannot be substituted merely because the source and target were recorded at one physical station.

This separation is proposed to prevent the phrase “360 formal trials” from silently changing the active capture-basin design. It is not a scientific or execution authorization.

## Proposed result-integrity contract

Every future trial would bind all of the following through a new authenticated formal lock:

- formal trial ID, scene, station, snapshot, track, and backend;
- byte-level source and target SHA256 values;
- the exact 4×4 `T0` supplied as `T_reference` to the frozen backend contract;
- backend parameter-contract SHA256 `6a1ebdee6b34390f1430eab371e1c4108db1f124b59b7239d74785efa6474af9`;
- finite 4×4 estimate, finite outcome metrics, runtime, and correspondence counts;
- formal-versus-fixture provenance and publication eligibility.

Schema validation alone never grants authority. A fixture may test serialization and rejection behavior, but it must have `fixture_only=true`, `publish_eligible=false`, and `formal_authority=false`; a publication request for any fixture must fail.

## Activation requirements

Activation would require every item below in a later, explicitly authorized change:

1. W04 is acquired with real field metadata, three passing stations, and no fabricated location.
2. W04 passes registration-free WEAK admission, and the batch is re-frozen and independently verified as exactly 3 RICH + 3 WEAK scenes, 18 stations, and 180 snapshots.
3. A versioned preregistration authority explicitly approves the zero-perturbation mainline and identifies how it coexists with the capture-basin track.
4. A new anchor, formal lock, and fingerprint bind every allowed trial and exact `T0`.
5. The formal result schema and validator tests pass without importing or invoking a backend.
6. A separate explicit registration authorization is issued.

Until all six occur, this file remains `PROPOSED_NOT_ACTIVE` and has `FORMAL_AUTHORITY=false`.
