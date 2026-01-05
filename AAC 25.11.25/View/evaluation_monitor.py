# ============================================================
# evaluation_monitor.py
# ============================================================
# Live Evaluation and Performance Monitor for AACScreen.py
# Logs each AAC event (input, normalization, category detection, latency)
# Computes rolling statistics, writes logs to CSV, and optionally
# displays live metrics overlay on the Kivy interface.
# ============================================================

import csv
import time
import threading
from statistics import mean
from collections import deque

# Optional overlay dependencies (for Kivy UI display)
try:
    from kivy.clock import Clock
    from kivy.uix.label import Label
    from kivy.uix.anchorlayout import AnchorLayout
except ImportError:
    # Allow headless (non-Kivy) execution
    Clock = Label = AnchorLayout = None


class EvaluationMonitor:
    """
    Periodically evaluates AAC system performance in real time.
    Logs all user events and computes live success rates, latencies,
    and can optionally display an on-screen overlay in Kivy.
    """

    def __init__(self, interval=120, logfile="live_metrics.csv", max_events=500):
        """
        :param interval: Evaluation update frequency (seconds)
        :param logfile: Path to CSV file storing logs
        :param max_events: Maximum recent events retained in memory
        """
        self.interval = interval
        self.logfile = logfile
        self.events = deque(maxlen=max_events)
        self.running = False
        self.overlay_label = None

    # ============================================================
    # Public Methods
    # ============================================================

    def start(self):
        """Starts the live evaluation thread."""
        self.running = True
        threading.Thread(target=self._periodic_eval, daemon=True).start()
        print(f"[EVAL] Live monitor started (interval={self.interval}s)")

    def stop(self):
        """Stops the evaluation thread."""
        self.running = False
        print("[EVAL] Live monitor stopped.")

    def log_event(self, input_text, normalized_text, predicted_category, success, response_time):
        """
        Logs an AAC event (input, normalized text, predicted category, success flag, latency)
        """
        event = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_text": input_text,
            "normalized_text": normalized_text,
            "predicted_category": predicted_category,
            "success": int(success),
            "response_time": round(response_time, 3)
        }

        # Keep in memory for quick rolling stats
        self.events.append(event)

        # Save to CSV
        self._write_to_csv(event)

    # ============================================================
    # CSV Logging
    # ============================================================

    def _write_to_csv(self, event):
        """Appends event to CSV log file (creates header if new)."""
        try:
            with open(self.logfile, "a", newline='', encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=event.keys())
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(event)
        except Exception as e:
            print(f"[EVAL LOG ERROR] {e}")

    # ============================================================
    # Periodic Evaluation Summary (Console Output)
    # ============================================================

    def _periodic_eval(self):
        """Runs continuously, printing evaluation summaries periodically."""
        while self.running:
            time.sleep(self.interval)
            if not self.events:
                continue

            success_rate = sum(e['success'] for e in self.events) / len(self.events) * 100
            avg_latency = mean(e['response_time'] for e in self.events)

            print(
                f"[EVAL] ✅ Success Rate: {success_rate:.2f}% | "
                f"⏱ Avg Latency: {avg_latency:.2f}s | "
                f"Samples: {len(self.events)}"
            )

            # Update Kivy overlay if active
            if self.overlay_label:
                try:
                    Clock.schedule_once(lambda dt: self._update_overlay(success_rate, avg_latency))
                except Exception:
                    pass

    # ============================================================
    # Optional Kivy Overlay for Real-Time Metrics
    # ============================================================

    def attach_dashboard_overlay(self, root_layout):
        """
        Attaches a live metrics overlay to the top-right of a Kivy layout.
        Updates automatically every 10 seconds.
        """
        if not Label or not AnchorLayout:
            print("[EVAL] Kivy overlay unavailable (Kivy not imported).")
            return

        overlay = AnchorLayout(anchor_x='right', anchor_y='top', size_hint=(None, None), size=(250, 100))
        lbl = Label(
            text="Eval: Initializing...",
            color=(1, 1, 1, 1),
            size_hint=(None, None),
            size=(250, 100),
            font_size='18sp',
            halign='left',
            valign='middle'
        )
        overlay.add_widget(lbl)
        root_layout.add_widget(overlay)
        self.overlay_label = lbl

        # Update overlay text periodically
        Clock.schedule_interval(lambda dt: self._refresh_overlay(), 10)
        print("[EVAL] Dashboard overlay attached to AAC UI.")

    def _refresh_overlay(self):
        """Recomputes metrics and updates overlay label."""
        if not self.events:
            return
        success_rate = sum(e['success'] for e in self.events) / len(self.events) * 100
        avg_latency = mean(e['response_time'] for e in self.events)
        self._update_overlay(success_rate, avg_latency)

    def _update_overlay(self, success_rate, avg_latency):
        """Updates the overlay text and color dynamically."""
        if not self.overlay_label:
            return
        self.overlay_label.text = (
            f"✅ {success_rate:.1f}%  ⏱ {avg_latency:.2f}s\\n"
            f"📊 {len(self.events)} samples"
        )
        if success_rate >= 85:
            self.overlay_label.color = (0.8, 1.0, 0.8, 1)
        elif success_rate >= 70:
            self.overlay_label.color = (1.0, 1.0, 0.6, 1)
        else:
            self.overlay_label.color = (1.0, 0.7, 0.7, 1)


# ============================================================
# Example Usage inside AACScreen
# ============================================================
# self.evaluator = EvaluationMonitor(interval=120)
# self.evaluator.start()
#
# start_time = time.time()
# ... (your processing)
# elapsed = time.time() - start_time
# self.evaluator.log_event(
#     input_text=text,
#     normalized_text=norm_text,
#     predicted_category=matched_category or "None",
#     success=True if matched_category else False,
#     response_time=elapsed
# )
#
# # Optional Kivy overlay (in build_ui)
# self.evaluator.attach_dashboard_overlay(root)
# ============================================================
