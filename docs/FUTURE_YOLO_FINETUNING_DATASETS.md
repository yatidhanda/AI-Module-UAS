# Future YOLO Fine-Tuning Datasets

This roadmap is documentation only. The current VisDrone work adds sequence testing and does not implement detector fine-tuning.

GarudaEdge remains a non-weaponized, human-in-the-loop ISR/search/monitoring demo. Future dataset work must remain limited to detection, classification, confidence scoring, evidence packaging, and human review.

## Candidate Public Datasets

- VisDrone: aerial civilian object detection and tracking across vehicles, pedestrians, bicycles, buses, and motorcycles.
- UAVDT: vehicle-heavy UAV detection and tracking, useful for road and traffic scenarios.
- Stanford Drone Dataset: campus-style overhead pedestrian and cyclist motion.
- Okutama-Action: aerial human movement/action stress testing.
- Public Roboflow Universe datasets: may later be evaluated for specialized object detection if licensing and safety constraints permit.

## Safety Boundary

Do not implement autonomous engagement, weapon release, lethal targeting, attack planning, or final high-consequence authorization.

Any future defence-object fine-tuning must remain limited to detection, classification, confidence scoring, evidence packaging, and human review.

The policy engine must remain the final safety gate and `human_review_required` must remain true.
