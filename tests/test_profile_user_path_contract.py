from tools import profile_user_path


def test_expected_surface_publications_follow_solver_frame_cadence():
    frames = [{"solver_frame": frame} for frame in range(1, 7)]

    assert profile_user_path._expected_surface_publications(frames, 2) == 4
    assert profile_user_path._expected_surface_publications(frames, 1) == 6


def test_expected_surface_publications_handle_empty_measurement():
    assert profile_user_path._expected_surface_publications([], 2) == 0


def test_warm_rows_drop_lazy_compilation_frames_from_six_frame_profile():
    rows = [{"frame": frame} for frame in range(6)]

    assert profile_user_path._warm_rows(rows) == rows[3:]
    assert profile_user_path._warm_rows(rows[:3]) == rows[:3]
