# emotion_engine.py
import joblib
import librosa
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")


class EmotionEngine:
    def __init__(self, model_path="emoq_emotion_model.joblib"):
        print("🎭 Loading EmoQ Emotion Model...")
        self.model = None
        self.bundle = None
        self.feature_cols = []
        self.abstention = {}

        if not os.path.exists(model_path):
            print(f"❌ Error: Model file '{model_path}' not found.")
            return

        try:
            self.bundle = joblib.load(model_path)
            self.model = self.bundle["model"]
            self.feature_cols = self.bundle["feature_cols"]
            self.abstention = self.bundle.get("abstention", {})
            print("✅ Emotion Model Loaded!")
        except Exception as e:
            print(f"❌ Failed to load model bundle: {e}")
            self.model = None

        # Match training defaults
        self.target_sr = 16000
        self.trim_top_db = 25
        self.n_mfcc = 40
        self.n_mels = 64
        self.fmax = 8000

        # Conservative live-audio rules
        self.min_duration_for_emotion = 0.70
        self.min_voiced_rate_for_emotion = 0.30
        self.min_log_rms_for_emotion = -5.5
        self.max_zcr_for_stable_emotion = 0.18

        # Stronger protection against false live predictions
        self.sad_extra_margin = 0.35
        self.angry_extra_margin = 0.15
        self.happy_extra_margin = 0.15

    def load_audio(self, path):
        y, sr = librosa.load(path, sr=self.target_sr, mono=True)
        y, _ = librosa.effects.trim(y, top_db=self.trim_top_db)

        if y is None or len(y) == 0:
            y = np.zeros(int(0.5 * self.target_sr), dtype=np.float32)

        # Keep real amplitude information for RMS-based gating.
        y = y.astype(np.float32)
        return y, sr

    def _add_stats(self, features: dict, prefix: str, M: np.ndarray):
        m = np.mean(M, axis=1)
        s = np.std(M, axis=1)
        for i in range(len(m)):
            features[f"{prefix}{i+1}_mean"] = float(m[i])
            features[f"{prefix}{i+1}_std"] = float(s[i])

    def extract_features(self, path):
        try:
            y, sr = self.load_audio(path)

            if len(y) < 100:
                return None

            duration_sec = float(len(y) / sr)

            # Prosody / pitch
            try:
                f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
                f0 = f0[np.isfinite(f0)]
            except Exception:
                f0 = np.array([], dtype=np.float32)

            if len(f0) > 0:
                voiced = f0[(f0 >= 60) & (f0 <= 350)]
                voiced_rate = float(len(voiced) / len(f0))
                f0_use = voiced if len(voiced) >= 3 else f0

                f0_mean = float(np.mean(f0_use))
                f0_std = float(np.std(f0_use))
                f0_median = float(np.median(f0_use))
                f0_min = float(np.min(f0_use))
                f0_max = float(np.max(f0_use))
            else:
                voiced_rate = 0.0
                f0_mean = f0_std = f0_median = f0_min = f0_max = 0.0

            # Energy
            rms = librosa.feature.rms(y=y)[0]
            log_rms = np.log(rms + 1e-8)
            log_rms_mean = float(np.mean(log_rms))
            log_rms_std = float(np.std(log_rms))

            # Spectral basics
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
            zcr = librosa.feature.zero_crossing_rate(y)[0]

            # Spectral contrast
            S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) + 1e-9
            contrast = librosa.feature.spectral_contrast(S=S, sr=sr)

            # Voice quality proxy
            y_harm, y_perc = librosa.effects.hpss(y)
            harm_rms = float(np.mean(librosa.feature.rms(y=y_harm)[0]))
            perc_rms = float(np.mean(librosa.feature.rms(y=y_perc)[0]))
            hpss_ratio = float(harm_rms / (perc_rms + 1e-9))

            features = {
                "duration_sec": duration_sec,

                "voiced_rate": voiced_rate,
                "f0_mean": f0_mean,
                "f0_std": f0_std,
                "f0_median": f0_median,
                "f0_min": f0_min,
                "f0_max": f0_max,

                "log_rms_mean": log_rms_mean,
                "log_rms_std": log_rms_std,

                "centroid_mean": float(np.mean(centroid)),
                "centroid_std": float(np.std(centroid)),
                "bandwidth_mean": float(np.mean(bandwidth)),
                "bandwidth_std": float(np.std(bandwidth)),
                "rolloff_mean": float(np.mean(rolloff)),
                "rolloff_std": float(np.std(rolloff)),
                "zcr_mean": float(np.mean(zcr)),
                "zcr_std": float(np.std(zcr)),

                "contrast_mean": float(np.mean(contrast)),
                "contrast_std": float(np.std(contrast)),

                "hpss_ratio": hpss_ratio,
            }

            # MFCC family
            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=self.n_mels, fmax=self.fmax
            )
            logmel = librosa.power_to_db(mel, ref=np.max)

            mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=self.n_mfcc)
            d1 = librosa.feature.delta(mfcc, order=1)
            d2 = librosa.feature.delta(mfcc, order=2)

            self._add_stats(features, "mfcc", mfcc)
            self._add_stats(features, "d1_mfcc", d1)
            self._add_stats(features, "d2_mfcc", d2)

            return features

        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def _scores_to_pred_and_margin(self, X):
        scores = self.model.decision_function(X)
        classes = self.model.classes_.astype(str)

        if scores.ndim == 1:
            pred = self.model.predict(X).astype(str)
            margins = np.abs(scores)
            score_dicts = [{"score": float(s)} for s in scores]
            return pred, margins, score_dicts

        top1_idx = np.argmax(scores, axis=1)
        pred_labels = classes[top1_idx]

        part = np.argpartition(scores, -2, axis=1)[:, -2:]
        top2_scores = np.take_along_axis(scores, part, axis=1)
        top2_sorted = np.sort(top2_scores, axis=1)
        margins = (top2_sorted[:, 1] - top2_sorted[:, 0]).astype(float)

        score_dicts = []
        for i in range(scores.shape[0]):
            d = {classes[j]: float(scores[i, j]) for j in range(scores.shape[1])}
            score_dicts.append(d)

        return pred_labels.astype(str), margins, score_dicts

    def apply_abstention(self, pred_labels, margins, margin_threshold, fallback_label):
        pred_labels = np.array(pred_labels, dtype=str)
        margins = np.array(margins, dtype=float)
        abstain_mask = margins < margin_threshold
        out = pred_labels.copy()
        out[abstain_mask] = fallback_label
        return out, abstain_mask

    def get_clip_quality_flags_from_features(self, X_row: pd.DataFrame):
        row = X_row.iloc[0]

        duration_sec = float(row.get("duration_sec", 0.0))
        voiced_rate = float(row.get("voiced_rate", 0.0))
        log_rms_mean = float(row.get("log_rms_mean", -999.0))
        zcr_mean = float(row.get("zcr_mean", 999.0))

        flags = {
            "too_short": duration_sec < self.min_duration_for_emotion,
            "too_unvoiced": voiced_rate < self.min_voiced_rate_for_emotion,
            "too_quiet": log_rms_mean < self.min_log_rms_for_emotion,
            "too_noisy_or_unstable": zcr_mean > self.max_zcr_for_stable_emotion,
        }
        flags["bad_for_emotion"] = any(flags.values())
        return flags

    def adjust_prediction_with_live_rules(self, pred, margin, quality_flags, fallback_label="neutral"):
        if quality_flags.get("bad_for_emotion", False):
            return fallback_label, "audio_quality_gate"

        if pred == "sad" and margin < self.sad_extra_margin:
            return fallback_label, "sad_requires_stronger_margin"

        if pred == "angry" and margin < self.angry_extra_margin:
            return fallback_label, "angry_requires_stronger_margin"

        if pred == "happy" and margin < self.happy_extra_margin:
            return fallback_label, "happy_requires_stronger_margin"

        return pred, "accepted"

    def predict(self, audio_path, debug=True):
        if self.model is None:
            return "neutral"

        feat_dict = self.extract_features(audio_path)
        if feat_dict is None:
            return "neutral"

        row = {c: feat_dict.get(c, 0.0) for c in self.feature_cols}
        X = pd.DataFrame([row], columns=self.feature_cols)
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True))
        X = X.fillna(0.0)

        try:
            fallback = str(self.abstention.get("fallback_label", "neutral"))
            margin_thr = float(self.abstention.get("tuned_margin", 0.0))

            pred_labels, margins, score_dicts = self._scores_to_pred_and_margin(X)
            raw_pred = str(pred_labels[0])
            margin = float(margins[0])
            scores = score_dicts[0]

            pred_abstain, abstain_mask = self.apply_abstention(
                [raw_pred], [margin], margin_threshold=margin_thr, fallback_label=fallback
            )
            post_abstain_pred = str(pred_abstain[0])

            quality_flags = self.get_clip_quality_flags_from_features(X)
            final_pred, reason = self.adjust_prediction_with_live_rules(
                pred=post_abstain_pred,
                margin=margin,
                quality_flags=quality_flags,
                fallback_label=fallback
            )

            if debug:
                print("\n🎭 Emotion Debug")
                print(f"raw_pred           : {raw_pred}")
                print(f"post_abstain_pred  : {post_abstain_pred}")
                print(f"final_pred         : {final_pred}")
                print(f"margin             : {margin:.4f}")
                print(f"margin_threshold   : {margin_thr:.4f}")
                print(f"reason             : {reason}")
                print(f"quality_flags      : {quality_flags}")
                print(f"scores             : {scores}")

            return str(final_pred)

        except Exception as e:
            print(f"Prediction error: {e}")
            return "neutral"