from __future__ import annotations

from tensordev_native_cpu import library_path, registrations, type_registrations


def test_loader_retains_library_and_capsules():
    first = registrations()
    second = registrations()

    assert library_path().is_file()
    assert set(first) == {
        "tensordev_cpu_sym_horner_f32_v2",
        "tensordev_cpu_sym_horner_f64_v2",
    }
    assert first.keys() == second.keys()
    for name in first:
        assert first[name] is second[name]

    first_types = type_registrations()
    second_types = type_registrations()
    assert set(first_types) == {"tensordev.ragged_horner_state.v1"}
    assert first_types.keys() == second_types.keys()
    for name in first_types:
        assert first_types[name] is second_types[name]
