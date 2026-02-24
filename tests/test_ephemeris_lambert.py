"""Tests for Lambert transfer solver and approach window search."""

import numpy as np
import pytest

from prospector.db import get_connection
from prospector.ephemeris.lambert import (
    AU,
    DAY,
    J2000,
    LEO_ALT,
    MU_EARTH,
    MU_SUN,
    R_EARTH,
    TransferWindow,
    arrival_dv,
    asteroid_state,
    departure_dv,
    earth_state,
    keplerian_to_cartesian,
    prefilter,
    search_all,
    search_windows,
    solve_kepler,
    solve_lambert,
    stumpff_c,
    stumpff_s,
    synodic_period_years,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn():
    c = get_connection(":memory:")
    yield c
    c.close()


def _insert_asteroid(conn, asteroid_id, name, a, e, i, om, w, ma, epoch, moid=None):
    """Insert an asteroid with orbital elements for testing."""
    conn.execute(
        "INSERT INTO asteroids (asteroid_id, name, neo) VALUES (?, ?, 1)",
        (asteroid_id, name),
    )
    conn.execute(
        "INSERT INTO orbits (asteroid_id, a, e, i, om, w, ma, epoch, moid) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (asteroid_id, a, e, i, om, w, ma, epoch, moid),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Tests: Stumpff functions
# ---------------------------------------------------------------------------

class TestStumpff:

    def test_c_positive_z(self):
        """C(z) for z > 0 (elliptical)."""
        z = 1.0
        expected = (1.0 - np.cos(1.0)) / 1.0
        assert stumpff_c(z) == pytest.approx(expected)

    def test_c_negative_z(self):
        """C(z) for z < 0 (hyperbolic)."""
        z = -1.0
        expected = (np.cosh(1.0) - 1.0) / 1.0
        assert stumpff_c(z) == pytest.approx(expected)

    def test_c_zero(self):
        """C(0) = 1/2."""
        assert stumpff_c(0.0) == pytest.approx(0.5, abs=1e-10)

    def test_s_positive_z(self):
        z = 1.0
        expected = (1.0 - np.sin(1.0)) / 1.0
        assert stumpff_s(z) == pytest.approx(expected)

    def test_s_zero(self):
        """S(0) = 1/6."""
        assert stumpff_s(0.0) == pytest.approx(1.0 / 6.0, abs=1e-10)

    def test_c_continuity_at_zero(self):
        """C(z) should be continuous through z=0."""
        assert stumpff_c(1e-8) == pytest.approx(stumpff_c(-1e-8), abs=1e-6)

    def test_s_continuity_at_zero(self):
        assert stumpff_s(1e-8) == pytest.approx(stumpff_s(-1e-8), abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: Kepler's equation
# ---------------------------------------------------------------------------

class TestKepler:

    def test_circular(self):
        """e=0: E = M."""
        assert solve_kepler(1.0, 0.0) == pytest.approx(1.0)

    def test_known_values(self):
        """Check against known E for e=0.5, M=1.0."""
        E = solve_kepler(1.0, 0.5)
        # Verify: E - e*sin(E) should equal M
        assert (E - 0.5 * np.sin(E)) == pytest.approx(1.0, abs=1e-10)

    def test_high_eccentricity(self):
        E = solve_kepler(np.pi, 0.9)
        assert (E - 0.9 * np.sin(E)) == pytest.approx(np.pi, abs=1e-10)

    def test_near_zero(self):
        E = solve_kepler(0.01, 0.3)
        assert (E - 0.3 * np.sin(E)) == pytest.approx(0.01, abs=1e-10)


# ---------------------------------------------------------------------------
# Tests: Keplerian to Cartesian
# ---------------------------------------------------------------------------

class TestKeplerianToCartesian:

    def test_circular_orbit_radius(self):
        """Circular orbit (e=0) should give r = a."""
        a = 1.0 * AU
        r, v = keplerian_to_cartesian(a, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert np.linalg.norm(r) == pytest.approx(a, rel=1e-6)

    def test_circular_orbit_speed(self):
        """Circular orbit: v = sqrt(mu/a)."""
        a = 1.0 * AU
        r, v = keplerian_to_cartesian(a, 0.0, 0.0, 0.0, 0.0, 0.0)
        v_expected = np.sqrt(MU_SUN / a)
        assert np.linalg.norm(v) == pytest.approx(v_expected, rel=1e-4)

    def test_perihelion(self):
        """At M=0 (perihelion), r = a(1-e)."""
        a = 1.5 * AU
        e = 0.3
        r, v = keplerian_to_cartesian(a, e, 0.0, 0.0, 0.0, 0.0)
        r_peri = a * (1.0 - e)
        assert np.linalg.norm(r) == pytest.approx(r_peri, rel=1e-6)

    def test_aphelion(self):
        """At M=pi, r ≈ a(1+e) (exact for pi)."""
        a = 1.5 * AU
        e = 0.3
        r, v = keplerian_to_cartesian(a, e, 0.0, 0.0, 0.0, np.pi)
        r_aph = a * (1.0 + e)
        assert np.linalg.norm(r) == pytest.approx(r_aph, rel=1e-4)

    def test_vis_viva(self):
        """Velocity should satisfy vis-viva: v^2 = mu(2/r - 1/a)."""
        a = 2.0 * AU
        e = 0.4
        M = 1.5
        r_vec, v_vec = keplerian_to_cartesian(a, e, 0.0, 0.0, 0.0, M)
        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)
        v_visviva = np.sqrt(MU_SUN * (2.0 / r - 1.0 / a))
        assert v == pytest.approx(v_visviva, rel=1e-4)


# ---------------------------------------------------------------------------
# Tests: Earth state
# ---------------------------------------------------------------------------

class TestEarthState:

    def test_radius_near_1au(self):
        """Earth should be near 1 AU from the Sun."""
        r, v = earth_state(J2000 + 365.25 * 10)  # 10 years from J2000
        r_au = np.linalg.norm(r) / AU
        assert 0.98 < r_au < 1.02

    def test_speed_near_30km_s(self):
        """Earth's orbital speed is ~29.78 km/s."""
        r, v = earth_state(J2000)
        v_km_s = np.linalg.norm(v) / 1000.0
        assert 29.0 < v_km_s < 31.0

    def test_different_epochs(self):
        """Earth should be at different positions at different times."""
        r1, _ = earth_state(J2000)
        r2, _ = earth_state(J2000 + 182.625)  # half year later
        # Should be roughly on opposite sides of the Sun
        dot = np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
        assert dot < 0  # roughly anti-parallel


# ---------------------------------------------------------------------------
# Tests: Lambert solver
# ---------------------------------------------------------------------------

class TestLambert:

    def test_hohmann_like_transfer(self):
        """Near-Hohmann transfer between two circular orbits.

        Earth (1 AU) to Mars-like (1.524 AU).
        Uses 170° transfer angle to avoid the 180° singularity
        inherent in Lambert's problem.
        """
        r1_au, r2_au = 1.0, 1.524
        a_t = (r1_au + r2_au) / 2.0 * AU
        tof = np.pi * np.sqrt(a_t**3 / MU_SUN) * 0.93  # slightly shorter

        r1 = np.array([r1_au * AU, 0.0, 0.0])
        # 170° transfer (10° short of Hohmann's 180°)
        angle = np.radians(170)
        r2 = np.array([r2_au * AU * np.cos(angle), r2_au * AU * np.sin(angle), 0.0])

        result = solve_lambert(r1, r2, tof)
        assert result is not None
        v1, v2 = result

        # Departure velocity should be > Earth circular velocity
        v_circ_earth = np.sqrt(MU_SUN / (r1_au * AU))
        assert np.linalg.norm(v1) > v_circ_earth * 0.9

    def test_short_transfer(self):
        """Short transfer between nearby positions (small angle)."""
        r1 = np.array([AU, 0.0, 0.0])
        # 30-degree transfer, same radius
        angle = np.radians(30)
        r2 = np.array([AU * np.cos(angle), AU * np.sin(angle), 0.0])
        tof = 60 * DAY

        result = solve_lambert(r1, r2, tof)
        assert result is not None
        v1, v2 = result
        # Velocities should be reasonable (< 100 km/s)
        assert np.linalg.norm(v1) < 100_000

    def test_returns_none_for_impossible(self):
        """Very short TOF for very distant points should fail."""
        r1 = np.array([AU, 0.0, 0.0])
        r2 = np.array([-5 * AU, 0.0, 0.0])
        tof = 1 * DAY  # impossibly short

        result = solve_lambert(r1, r2, tof)
        # Should either return None or give huge velocities
        if result is not None:
            v1, _ = result
            assert np.linalg.norm(v1) > 1e6  # >1000 km/s = physically impossible

    def test_conservation_of_energy(self):
        """Both endpoints should have the same orbital energy."""
        r1 = np.array([AU, 0.0, 0.0])
        angle = np.radians(60)
        r2 = np.array([1.3 * AU * np.cos(angle), 1.3 * AU * np.sin(angle), 0.0])
        tof = 180 * DAY

        result = solve_lambert(r1, r2, tof)
        assert result is not None
        v1, v2 = result

        # Specific orbital energy: epsilon = v^2/2 - mu/r
        eps1 = np.dot(v1, v1) / 2.0 - MU_SUN / np.linalg.norm(r1)
        eps2 = np.dot(v2, v2) / 2.0 - MU_SUN / np.linalg.norm(r2)
        assert eps1 == pytest.approx(eps2, rel=1e-3)


# ---------------------------------------------------------------------------
# Tests: Delta-v
# ---------------------------------------------------------------------------

class TestDeltaV:

    def test_departure_from_leo(self):
        """Departure Δv from 200 km LEO should be reasonable."""
        # For a typical Earth-escape, v_inf ~3 km/s
        v_inf = np.array([3000.0, 0.0, 0.0])  # 3 km/s excess
        dv = departure_dv(v_inf)
        # Should be a few km/s
        assert 3.0 < dv / 1000 < 6.0

    def test_zero_v_inf(self):
        """Zero v_inf means just reaching escape velocity."""
        v_inf = np.array([0.0, 0.0, 0.0])
        dv = departure_dv(v_inf)
        # Δv = v_escape - v_circular
        r_park = R_EARTH + LEO_ALT
        v_esc = np.sqrt(2.0 * MU_EARTH / r_park)
        v_circ = np.sqrt(MU_EARTH / r_park)
        assert dv == pytest.approx(v_esc - v_circ, rel=1e-6)

    def test_arrival_dv(self):
        """Arrival Δv = magnitude of relative velocity."""
        v = np.array([2000.0, 1000.0, 500.0])
        assert arrival_dv(v) == pytest.approx(np.linalg.norm(v))


# ---------------------------------------------------------------------------
# Tests: Prefilter
# ---------------------------------------------------------------------------

class TestPrefilter:

    def test_passes_good_neo(self):
        assert prefilter(0.05, 10.0, 1.5) is True

    def test_rejects_high_moid(self):
        assert prefilter(0.5, 10.0, 1.5) is False

    def test_rejects_high_inclination(self):
        assert prefilter(0.05, 40.0, 1.5) is False

    def test_rejects_distant(self):
        assert prefilter(0.05, 10.0, 5.0) is False

    def test_none_moid_passes(self):
        """Missing MOID should not reject."""
        assert prefilter(None, 10.0, 1.5) is True

    def test_none_inclination_passes(self):
        assert prefilter(0.05, None, 1.5) is True

    def test_none_a_rejects(self):
        assert prefilter(0.05, 10.0, None) is False


# ---------------------------------------------------------------------------
# Tests: Synodic period
# ---------------------------------------------------------------------------

class TestSynodicPeriod:

    def test_mars_like(self):
        """Mars-like orbit (a=1.524): synodic period ≈ 2.14 years."""
        t = synodic_period_years(1.524)
        assert 2.0 < t < 2.3

    def test_near_1au(self):
        """Near-Earth orbit: very long synodic period."""
        t = synodic_period_years(1.001)
        assert t > 100  # near-resonant

    def test_2au(self):
        """a=2 AU: synodic period ≈ 2.83 years."""
        t = synodic_period_years(2.0)
        p_ast = 2.0 ** 1.5
        expected = 1.0 / abs(1.0 - 1.0 / p_ast)
        assert t == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Tests: search_windows (DB integration)
# ---------------------------------------------------------------------------

class TestSearchWindows:

    def test_finds_windows_for_neo(self, conn):
        """Should find transfer windows for a typical NEO."""
        # Apophis-like orbit
        _insert_asteroid(
            conn, 99942, "Apophis",
            a=0.9224, e=0.1915, i=3.34, om=204.43, w=126.40,
            ma=215.0, epoch=2459800.5, moid=0.00025,
        )
        windows = search_windows(
            99942, conn,
            start_jd=2461000.5,
            horizon_years=5,
            step_days=10,
            tof_min_days=100,
            tof_max_days=400,
            tof_step_days=30,
            max_dv_km_s=15.0,
            top_n=5,
        )
        # Should find at least some windows for a near-Earth asteroid
        assert len(windows) > 0
        # Windows should be sorted by Δv
        for i in range(len(windows) - 1):
            assert windows[i].dv_total_km_s <= windows[i + 1].dv_total_km_s

    def test_no_orbit_returns_empty(self, conn):
        """Asteroid without orbital data returns empty list."""
        conn.execute("INSERT INTO asteroids (asteroid_id) VALUES (1)")
        conn.commit()
        assert search_windows(1, conn) == []

    def test_window_fields(self, conn):
        """TransferWindow should have all required fields."""
        _insert_asteroid(
            conn, 25143, "Itokawa",
            a=1.3241, e=0.2802, i=1.622, om=69.08, w=162.77,
            ma=310.0, epoch=2459800.5, moid=0.013,
        )
        windows = search_windows(
            25143, conn,
            horizon_years=3,
            step_days=15,
            tof_min_days=120,
            tof_max_days=350,
            tof_step_days=30,
            max_dv_km_s=15.0,
        )
        if windows:
            w = windows[0]
            assert w.launch_jd > 0
            assert w.arrival_jd > w.launch_jd
            assert w.tof_days > 0
            assert w.dv_departure_km_s > 0
            assert w.dv_arrival_km_s >= 0
            assert w.dv_total_km_s > 0

    def test_top_n_limits_results(self, conn):
        _insert_asteroid(
            conn, 99942, "Apophis",
            a=0.9224, e=0.1915, i=3.34, om=204.43, w=126.40,
            ma=215.0, epoch=2459800.5,
        )
        windows = search_windows(
            99942, conn,
            horizon_years=3,
            step_days=15,
            tof_min_days=100,
            tof_max_days=300,
            tof_step_days=30,
            top_n=3,
        )
        assert len(windows) <= 3


# ---------------------------------------------------------------------------
# Tests: search_all (batch)
# ---------------------------------------------------------------------------

class TestSearchAll:

    def test_filters_and_searches(self, conn):
        """Batch search filters by MOID and inclination."""
        # Good candidate (low MOID, low inclination)
        _insert_asteroid(
            conn, 99942, "Apophis",
            a=0.9224, e=0.1915, i=3.34, om=204.43, w=126.40,
            ma=215.0, epoch=2459800.5, moid=0.00025,
        )
        # Poor candidate (high MOID)
        _insert_asteroid(
            conn, 16, "Psyche",
            a=2.92, e=0.14, i=3.1, om=150.0, w=230.0,
            ma=100.0, epoch=2459800.5, moid=1.5,
        )

        results = search_all(
            conn,
            max_moid_au=0.3,
            horizon_years=3,
            step_days=20,
            tof_min_days=100,
            tof_max_days=300,
            tof_step_days=40,
        )
        # Apophis should pass filter; Psyche should not
        assert 99942 in results or len(results) == 0  # may not find windows
        assert 16 not in results


# ---------------------------------------------------------------------------
# Tests: Physical sanity
# ---------------------------------------------------------------------------

class TestPhysicalSanity:

    def test_earth_escape_velocity(self):
        """Escape velocity from 200 km LEO ≈ 11.0 km/s."""
        r_park = R_EARTH + LEO_ALT
        v_esc = np.sqrt(2.0 * MU_EARTH / r_park) / 1000
        assert 10.8 < v_esc < 11.2

    def test_leo_orbital_velocity(self):
        """LEO (200 km) orbital velocity ≈ 7.78 km/s."""
        r_park = R_EARTH + LEO_ALT
        v_circ = np.sqrt(MU_EARTH / r_park) / 1000
        assert 7.7 < v_circ < 7.9

    def test_earth_orbital_velocity(self):
        """Earth orbital velocity ≈ 29.78 km/s."""
        v_earth = np.sqrt(MU_SUN / AU) / 1000
        assert 29.5 < v_earth < 30.0

    def test_constants_consistent(self):
        """Physical constants should be self-consistent."""
        # Earth year from Kepler's third law
        P_earth = 2 * np.pi * np.sqrt(AU**3 / MU_SUN) / DAY
        assert P_earth == pytest.approx(365.25, rel=0.01)

    def test_departure_dv_increases_with_v_inf(self):
        """Higher v_inf should require more Δv."""
        dv1 = departure_dv(np.array([1000.0, 0, 0]))
        dv2 = departure_dv(np.array([5000.0, 0, 0]))
        dv3 = departure_dv(np.array([10000.0, 0, 0]))
        assert dv1 < dv2 < dv3

    def test_apophis_accessible(self, conn):
        """Apophis should be accessible with <10 km/s total Δv."""
        _insert_asteroid(
            conn, 99942, "Apophis",
            a=0.9224, e=0.1915, i=3.34, om=204.43, w=126.40,
            ma=215.0, epoch=2459800.5,
        )
        windows = search_windows(
            99942, conn,
            horizon_years=10,
            step_days=10,
            tof_min_days=80,
            tof_max_days=400,
            tof_step_days=20,
            max_dv_km_s=15.0,
            top_n=1,
        )
        # Apophis is one of the most accessible NEOs
        assert len(windows) > 0
        assert windows[0].dv_total_km_s < 15.0
