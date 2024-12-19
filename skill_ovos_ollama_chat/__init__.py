from datetime import datetime
from ovos_utils import classproperty
from ovos_utils.process_utils import RuntimeRequirements
from ovos_workshop.skills.fallback import FallbackSkill
from ollama import Client as ocli
import requests
import re
from langcodes import standardize_tag


class OllamaChatSkill(FallbackSkill):
    @classproperty
    def runtime_requirements(self):
        return RuntimeRequirements(
            internet_before_load=True,
            network_before_load=True,
            gui_before_load=False,
            requires_internet=True,
            requires_network=True,
            requires_gui=False,
            no_internet_fallback=False,
            no_network_fallback=False,
            no_gui_fallback=True,
        )

    def initialize(self):
        self.fallback_priority = self.settings.get("priority", 90)
        self.register_fallback(self.handle_fallback, int(self.fallback_priority))
        if self.settings.get("handle_utterance", False):
            self.add_event("recognizer_loop:utterance", self.handle_utterance)

        self.timestamp = datetime.now()
        self.reset_chat()
        self.doc_snippets = []
        self.settings_change_callback = self.on_settings_changed

    def on_settings_changed(self):
        self.url = self.settings.get("url")
        self.fasttext_url = self.settings.get("fasttext_url")
        self.max_tokens = self.settings.get("max_tokens", 150)

        self.context_timeout = self.settings.get("context_timout", 600)
        self.preamble = self.settings.get(
            "preamble", "Your name is Jarvis. You are a helpful language model."
        )
        self.chat_history = [{"role": "system", "message": self.preamble}]
        if self.settings.get("priority") != self.fallback_priority:
            self.log.info("Priority setting has changed. Resetting fallback")
            self.fallback_priority = self.settings.get("priority", 90)
            self.register_fallback(self.handle_fallback, self.fallback_priority)
        self.model = self.settings.get("model", "phi3")
        self.connectors = self.settings.get("search_connectors", [])
        self.ollama_connect()

    def reset_chat(self):
        self.on_settings_changed()

    def ollama_connect(self):
        try:
            self.ollama = ocli(self.url)
            self.log.info("Connected to Ollama.")
        except Exception as e:
            self.log.error("Failed to connect Ollama client. " f"Got exception: {e}")

    def detect_lang(self, text):
        self.log.debug(f"Detecting language for: {text}")
        try:
            resp = requests.get(f"{self.fasttext_url}/language_detect?text={text}")
            return standardize_tag(resp.json()[0][0])
        except requests.exceptions.RequestException as e:
            self.log.error("Lang detect error: %s", e)
            return "en"
        except KeyError as e:
            self.log.error("%s\nMissing lang in reply: %s", (e, resp.json()))
            return "en"

    def update_chat_history(self, role, message):
        self.chat_history.append({"role": role, "content": message})
        self.log.debug("Updated chat history: %s", message)

    def handle_utterance(self, message):
        utt = message.data.get("utterances")[0]
        try:
            if self.fasttext_url != "":
                lang = self.detect_lang(utt)
                message.context["lang"] = lang
                self.log.debug(f"Message context language set to {lang}")
        except Exception as e:
            self.log.error("Error using the language detector plugin: %s", e)

    def classify(self, msg):
        examples = self.for_me
        examples.extend(self.not_for_me)

    def chat(self):
        self.log.info("Sending to Ollama LLM: %s", self.model)
        try:
            return self.ollama.chat(
                model=self.model,
                messages=self.chat_history,
                keep_alive=-1,
                stream=True,
                options={
                    "num_predict": self.max_tokens,
                },
            )
        except Exception as e:
            self.log.error(f"Ollama chat api error: {e}")
            return False

    def converse(self, message):
        msg = message.data.get("utterances")[0]
        if msg is None:
            return False
        elif self.voc_match(msg, "thanks"):
            self.speak_dialog("noproblem")
            return True
        # is_relevant = self.classify(msg)
        # if is_relevant:
        #     self.process_stream(msg)
        # else:
        #     self.speak_dialog("iamdone")
        #     return True

    def process_stream(self, message):
        phrase = ""
        full_text = ""
        look_ahead = ""
        token_count = 0
        sentence_end = False
        token = ""
        time_lapse = datetime.now() - self.timestamp
        if time_lapse.seconds > self.context_timeout:
            self.log.debug(
                f"{self.context_timeout}s has run out since {self.timestamp}. "
                "Resetting chat."
            )
            self.reset_chat()

        self.update_chat_history("user", message.data["utterance"])
        self.log.debug(f"Chat History: {self.chat_history}")
        self.gui.show_page("Chat.qml", 90)
        try:
            for chunk in self.chat():
                self.log.debug(f"Received chunk: {chunk}")
                if chunk["message"]["content"] in ["\n", "\r\n"]:
                    continue
                look_ahead = chunk["message"]["content"]
                self.log.debug(f"Streaming from {self.model}: {look_ahead}")
                if look_ahead or chunk["done"]:
                    if token != "":
                        if (
                            "." in token
                            or "?" in token
                            or "!" in token
                            or "\n" in token
                        ):
                            if "." in token and bool(re.search("[0-9]", look_ahead)):
                                self.log.debug(
                                    "Skipping potential ordered list item cut."
                                )
                            else:
                                if token_count > 0:
                                    sentence_end = True

                        token_count = token_count + 1
                        phrase = phrase + token
                        if token_count > 20 or chunk["done"] or sentence_end:
                            full_text = f"{full_text}{phrase}"
                            self.gui["ChatText"] = full_text
                            self.log.debug(f"Speaking: {phrase}")
                            self.speak_dialog(
                                phrase,
                                expect_response=chunk["done"],
                                wait=chunk["done"],
                            )
                            self.update_chat_history("assistant", phrase)
                            token_count = 0
                            phrase = ""
                            sentence_end = False
                    token = look_ahead

            return True
        except Exception as e:
            self.log.error(f"Error while processing Ollama chat stream: {e}")
            self.log.debug(f"Last chunk: {chunk}")
            return False

    def handle_fallback(self, message):
        # Mic bug
        if message.data["utterance"] == "the":
            return False

        try:
            return self.process_stream(message)
        except Exception as e:
            self.log.error(f"Failed handling chat stream: {e}")
            return False

        return True


def create_skill():
    return OllamaChatSkill()


classifyNotForMe = [
    "Is the dinner ready?",
    "In today's news",
    "Hello, this is him",
    "This is her",
    "You see, I told you",
    "What did he say?",
    "What did she say?",
    "Get ready for bed",
    "I am getting some",
    "and he scores!",
]

classifyForMe = [
    "Are you sure",
    "That does not sound right",
    "Can you tell me more about that",
    "In what year was that",
    "Who was the second person",
    "Where did you get that info",
    "How did you deduce that",
    "Is that it",
]
