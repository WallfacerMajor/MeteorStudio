import threading
import time
import unittest

from background_tasks import BackgroundTaskScheduler


class BackgroundTaskSchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scheduler = BackgroundTaskScheduler(max_workers=2, thread_name_prefix="test-task")

    def tearDown(self) -> None:
        self.scheduler.shutdown(wait=True)

    def test_replacement_suppresses_stale_result(self):
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        results = []

        def old(token):
            started.set()
            release.wait(2)
            return "old"

        self.scheduler.submit("detect", old, on_result=results.append)
        self.assertTrue(started.wait(1))
        self.scheduler.submit(
            "detect", lambda token: "new",
            on_result=lambda value: (results.append(value), finished.set()),
        )
        self.assertTrue(finished.wait(1))
        release.set()
        self.scheduler.shutdown(wait=True)
        self.assertEqual(results, ["new"])

    def test_cancelled_task_sees_token_and_cannot_publish(self):
        started = threading.Event()
        release = threading.Event()
        observed = []
        results = []

        def work(token):
            started.set()
            release.wait(2)
            observed.append(token.cancelled)
            return "stale"

        self.scheduler.submit("prefetch", work, on_result=results.append)
        self.assertTrue(started.wait(1))
        self.scheduler.cancel("prefetch")
        release.set()
        self.scheduler.shutdown(wait=True)
        self.assertEqual(observed, [True])
        self.assertEqual(results, [])

    def test_current_error_is_reported_with_traceback(self):
        finished = threading.Event()
        errors = []

        def fail(_token):
            raise ValueError("broken")

        self.scheduler.submit(
            "worker", fail,
            on_error=lambda exc, details: (errors.append((exc, details)), finished.set()),
        )
        self.assertTrue(finished.wait(1))
        self.assertIsInstance(errors[0][0], ValueError)
        self.assertIn("ValueError: broken", errors[0][1])

    def test_finished_task_is_removed_from_active_count(self):
        finished = threading.Event()
        self.scheduler.submit(
            "fast", lambda _token: "done",
            on_result=lambda _value: finished.set(),
        )
        self.assertTrue(finished.wait(1))
        self.assertEqual(self.scheduler.active_count(), 0)

    def test_transient_channel_releases_its_current_token(self):
        finished = threading.Event()
        self.scheduler.submit(
            "worker:1", lambda _token: "done",
            on_result=lambda _value: finished.set(),
            replace=False, retain_current=False,
        )
        self.assertTrue(finished.wait(1))
        deadline = time.monotonic() + 1
        while self.scheduler.tracked_channel_count("worker:") and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(self.scheduler.tracked_channel_count("worker:"), 0)


if __name__ == "__main__":
    unittest.main()
