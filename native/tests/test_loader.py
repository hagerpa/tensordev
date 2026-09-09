from __future__ import annotations

def test_loader_retains_library_and_capsules(native_extension):
    first = native_extension.registrations()
    second = native_extension.registrations()

    assert native_extension.library_path().is_file()
    assert set(first) == {
        "tensordev_cpu_sym_horner_f32_v2",
        "tensordev_cpu_sym_horner_f64_v2",
    }
    assert first.keys() == second.keys()
    for name in first:
        assert first[name] is second[name]

    first_types = native_extension.type_registrations()
    second_types = native_extension.type_registrations()
    assert set(first_types) == {"tensordev.ragged_horner_state.v1"}
    assert first_types.keys() == second_types.keys()
    for name in first_types:
        assert first_types[name] is second_types[name]
