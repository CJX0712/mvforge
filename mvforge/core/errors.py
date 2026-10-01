"""错误码体系 E100~E500 (单码单义)."""


class MVForgeError(Exception):
    """MVForge 基类错误."""

    code = "E000"

    def __init__(self, message: str, code: str | None = None) -> None:
        self.code = code or self.code
        super().__init__(f"[{self.code}] {message}")


class ConfigError(MVForgeError):
    code = "E100"


class DataError(MVForgeError):
    code = "E200"


class RepresentationError(MVForgeError):
    code = "E300"


class FusionError(MVForgeError):
    code = "E400"


class RetrievalError(MVForgeError):
    code = "E500"
