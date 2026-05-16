from garuda_edge.ingestion.schemas import ConfidenceResult, GeoEstimate, ISREvent


def test_example_event_validates():
    event = ISREvent(
        event_id="EVT-1",
        event_type="track_update",
        timestamp_s=0,
        frame_id=1,
        track_id="T-0001",
        object_class="person",
        bbox_xyxy=[1, 2, 3, 4],
        detection_confidence=0.8,
        track_age_frames=2,
        track_stability=0.5,
        platform={"lat": 1},
        link={"packet_loss": 0},
        geo_estimate=GeoEstimate(estimated_lat=1, estimated_lon=2, geo_uncertainty_m=10, method="test"),
        confidence=ConfidenceResult(final_confidence=0.7, level="medium", reasons=[], degraded_flags=[]),
        audit_notes=["human review required"],
    )
    assert event.track_id == "T-0001"
