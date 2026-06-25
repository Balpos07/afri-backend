import torch
import time
import requests
import os
from transformers import pipeline
import librosa
from io import BytesIO

class MedicalVoicePipeline:
    def __init__(self):
        init_start = time.time()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.api_key = "sk_live_nn1Pot6DEv8CxsxMJ95htsyUn9QTke22cnJExpd1BYI"
        self.tts_url = "https://yarngpt.ai/api/v1/tts"

        print(f"Using device: {self.device}")

        # =========================
        # STAGE 1: Pidgin ASR MODEL LOAD
        # =========================
        asr_start = time.time()

        print("Loading Pidgin ASR model (asr-nigerian-pidgin/pidgin-wav2vec2-xlsr53)...")

        try:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

            offline_start = time.time()

            self.asr = pipeline(
                "automatic-speech-recognition",
                model="asr-nigerian-pidgin/pidgin-wav2vec2-xlsr53",
                device=self.device
            )

            print(f"Model loaded from local cache in {time.time() - offline_start:.2f}s")

        except Exception:
            print("Model not found locally, downloading from Hugging Face...")

            os.environ["TRANSFORMERS_OFFLINE"] = "0"

            download_start = time.time()

            self.asr = pipeline(
                "automatic-speech-recognition",
                model="asr-nigerian-pidgin/pidgin-wav2vec2-xlsr53",
                device=self.device
            )

            print(f"Model downloaded + loaded in {time.time() - download_start:.2f}s")

        print(f"STAGE 1 MODEL LOAD TIME: {time.time() - asr_start:.2f}s")

        # =========================
        # STAGE 2: KNOWLEDGE BASE LOAD
        # =========================
        kb_start = time.time()

        import json

        with open("knowledge_base_pidgin.json", "r", encoding="utf-8") as f:
            self.knowledge_base = json.load(f)

        print(f"Professional Knowledge Base loaded with {len(self.knowledge_base)} symptoms!")

        print(f"STAGE 2 KNOWLEDGE BASE LOAD TIME: {time.time() - kb_start:.2f}s")

        print(f"TOTAL INITIALIZATION TIME: {time.time() - init_start:.2f}s\n")

    def transcribe(self, audio_bytes):
        stage_start = time.time()

        # Audio Loading
        audio_load_start = time.time()

        audio, sr = librosa.load(BytesIO(audio_bytes), sr=16000)

        print(f"Audio preprocessing time: {time.time() - audio_load_start:.2f}s")

        # ASR Inference
        asr_inference_start = time.time()

        result = self.asr(audio)

        print(f"ASR inference time: {time.time() - asr_inference_start:.2f}s")

        total_stage = time.time() - stage_start

        print(f"TOTAL TRANSCRIPTION STAGE TIME: {total_stage:.2f}s")

        return result["text"].lower()

    def fuzzy_substring_score(self, query: str, text: str):
        import difflib

        query_len = len(query)
        text_len = len(text)

        if text_len <= query_len + 3:
            return difflib.SequenceMatcher(None, query, text).ratio()

        max_score = 0.0
        words = text.split()
        q_words = query.split()
        window_size = len(q_words)

        if window_size == 0:
            return 0.0

        for i in range(len(words) - window_size + 1):

            window_text = " ".join(words[i:i+window_size])

            score = difflib.SequenceMatcher(None, query, window_text).ratio()

            if score > max_score:
                max_score = score

            if i + window_size < len(words):
                window_text_plus = " ".join(words[i:i+window_size+1])

                score2 = difflib.SequenceMatcher(None, query, window_text_plus).ratio()

                if score2 > max_score:
                    max_score = score2

        return max_score

    def generate_response(self, user_text: str):
        stage_start = time.time()

        print(f"DEBUG: Professional Knowledge Lookup for: {user_text}")

        cleanup_start = time.time()

        for char in [".", ",", "?", "!", "'", '"']:
            user_text = user_text.replace(char, "")

        user_text = user_text.strip()

        print(f"Text cleanup time: {time.time() - cleanup_start:.2f}s")

        matching_start = time.time()

        best_match_response = None
        best_match_key = None
        best_match_symptom = None
        highest_score = 0.0

        for item in self.knowledge_base:

            symptom_key = item["symptom"].replace("_", " ").lower()

            pidgin_val = item["pidgin"]

            if isinstance(pidgin_val, str):
                pidgin_keys = [pidgin_val.lower()]
            else:
                pidgin_keys = [k.lower() for k in pidgin_val]

            # Exact Match
            for p_key in pidgin_keys:

                if p_key in user_text or symptom_key in user_text:

                    print(f"DEBUG: Exact match found for '{p_key}'")
                    print(f"Knowledge matching time: {time.time() - matching_start:.2f}s")
                    print(f"TOTAL RESPONSE GENERATION TIME: {time.time() - stage_start:.2f}s")

                    matched_phrase = p_key if p_key in user_text else symptom_key
                    return item["recommendation"], item["symptom"], matched_phrase

            # Fuzzy Match
            for p_key in pidgin_keys:

                score = self.fuzzy_substring_score(p_key, user_text)

                if score > highest_score:
                    highest_score = score
                    best_match_response = item["recommendation"]
                    best_match_key = p_key
                    best_match_symptom = item["symptom"]

        print(f"Knowledge matching time: {time.time() - matching_start:.2f}s")

        if highest_score >= 0.60:
            print(
                f"DEBUG: Fuzzy match success! "
                f"User said '{user_text}', matched '{best_match_key}' "
                f"with {highest_score*100:.1f}% accuracy."
            )

            print(f"TOTAL RESPONSE GENERATION TIME: {time.time() - stage_start:.2f}s")

            return best_match_response, best_match_symptom, best_match_key

        print(
            f"DEBUG: No match found. "
            f"Highest similarity was only {highest_score*100:.1f}% "
            f"for '{best_match_key}'."
        )

        print(f"TOTAL RESPONSE GENERATION TIME: {time.time() - stage_start:.2f}s")

        return "Oga sorry oo, I no too understand wetin you talk. Abeg, try talk am again well-well make I know how your body dey do you.", None, None

    def text_to_speech(self, text: str):
        stage_start = time.time()

        print("DEBUG: Calling YarnGPT API for response...")

        payload = {
            "text": text,
            "voice": "Adaora",
            "response_format": "wav"
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            request_start = time.time()

            response = requests.post(
                self.tts_url,
                json=payload,
                headers=headers,
                timeout=60
            )

            response.raise_for_status()

            print(f"TTS API request time: {time.time() - request_start:.2f}s")
            print(f"TOTAL TTS STAGE TIME: {time.time() - stage_start:.2f}s")

            return response.content, 24000

        except Exception as e:
            print(f"ERROR: YarnGPT API failed: {e}")
            print(f"TOTAL TTS STAGE TIME: {time.time() - stage_start:.2f}s")

            return b"", 24000

    def process_voice(self, audio_bytes):
        overall_start = time.time()

        print("\n" + "=" * 60)
        print("STARTING MEDICAL VOICE PIPELINE")
        print("=" * 60)

        # =========================
        # STAGE 1: TRANSCRIPTION
        # =========================
        stage1_start = time.time()

        text = self.transcribe(audio_bytes)

        stage1_time = time.time() - stage1_start

        print("\n" + "=" * 50)
        print(f"STAGE 1 (ASR OUTPUT): '{text}'")
        print(f"STAGE 1 TIME: {stage1_time:.2f}s")
        print("=" * 50)

        # =========================
        # STAGE 2: KNOWLEDGE LOOKUP
        # =========================
        stage2_start = time.time()

        response_text, matched_symptom, matched_phrase = self.generate_response(text)

        stage2_time = time.time() - stage2_start

        print("\n" + "=" * 50)
        print(f"STAGE 2 (KNOWLEDGE MATCH): '{response_text}'")
        print(f"STAGE 2 TIME: {stage2_time:.2f}s")
        print("=" * 50)

        # =========================
        # STAGE 3: TEXT TO SPEECH
        # =========================
        stage3_start = time.time()

        audio_output, sr = self.text_to_speech(response_text)

        stage3_time = time.time() - stage3_start

        print("\n" + "=" * 50)
        print("STAGE 3 (TEXT TO SPEECH COMPLETED)")
        print(f"STAGE 3 TIME: {stage3_time:.2f}s")
        print("=" * 50)

        # =========================
        # TOTAL TIMING
        # =========================
        total_time = time.time() - overall_start

        print("\n" + "=" * 60)
        print("PIPELINE PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Stage 1 - ASR Transcription : {stage1_time:.2f}s")
        print(f"Stage 2 - Knowledge Lookup : {stage2_time:.2f}s")
        print(f"Stage 3 - TTS Generation   : {stage3_time:.2f}s")
        print("-" * 60)
        print(f"TOTAL PROCESSING TIME      : {total_time:.2f}s")
        print("=" * 60 + "\n")

        return {
            "input_text": text,
            "response_text": response_text,
            "matched_symptom": matched_symptom,
            "matched_phrase": matched_phrase,
            "audio": audio_output,
            "sample_rate": sr
        }
