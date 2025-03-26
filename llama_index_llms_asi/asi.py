"""ASI LLM implementation."""

import os
from typing import Any, Optional

from llama_index.llms.openai_like import OpenAILike

DEFAULT_MODEL = "asi1-mini"


class ASI(OpenAILike):
    """
    ASI LLM - Integration for ASI models.
    
    Currently supported models:
    - asi1-mini
    
    Examples:
        `pip install llama-index-llms-asi`

        ```python
        from llama_index_llms_asi import ASI

        # Set up the ASI class with the required model and API key
        llm = ASI(model="asi1-mini", api_key="your_api_key")
        
        # Call the complete method with a query
        response = llm.complete("Explain the importance of AI")
        
        print(response)
        ```
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        api_base: str = "https://api.asi1.ai/v1",
        is_chat_model: bool = True,
        is_function_calling_model: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ASI LLM.

        Args:
            model (str): The ASI model to use. Defaults to "asi1-mini".
            api_key (Optional[str]): The API key to use. If None, the ASI_API_KEY
                environment variable will be used. Defaults to None.
            api_base (str): The base URL for the ASI API. Defaults to
                "https://api.asi1.ai/v1".
            is_chat_model (bool): Whether the model supports chat. Defaults to True.
            is_function_calling_model (bool): Whether the model supports function
                calling. Defaults to False.
            **kwargs (Any): Additional arguments to pass to the OpenAILike constructor.
        """
        api_key = api_key or os.environ.get("ASI_API_KEY", None)
        if api_key is None:
            raise ValueError(
                "ASI API key is required. Set it using the api_key parameter "
                "or the ASI_API_KEY environment variable."
            )

        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            is_chat_model=is_chat_model,
            is_function_calling_model=is_function_calling_model,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "ASI"
