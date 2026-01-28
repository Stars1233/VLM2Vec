from transformers import AutoProcessor

class Qwen2_5OmniEmbeddingProcessor:
    """
    轻量 wrapper：内部直接使用 AutoProcessor。
    """
    def __init__(self, base_processor):
        self.base = base_processor

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        base = AutoProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(base)

    def __call__(
        self,
        text: str | None = None,
        image: str | None = None,
        video: str | None = None,
        audio: str | None = None,
        *,
        fps: int | float | None = None,
        load_audio_from_video: bool = False,
        add_generation_prompt: bool = False,
        return_tensors: str = "pt",
        **kwargs,
    ):
        if text is None and image is None and video is None and audio is None:
            raise ValueError("At least one modality must be provided.")

        return self.base(
            text=text,
            images=image,
            videos=video,
            audio=audio,
            return_tensors=return_tensors,
            **kwargs,
        )
