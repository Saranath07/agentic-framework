import time
import json
import logging
import datetime
import os
from typing import Optional, Dict, Any, List, Union
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from openai import OpenAI
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict

from dotenv import load_dotenv

load_dotenv()

# Configure logging
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(log_directory, "llm_api_calls.log")

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LLM_API")

# Set up JSON logger for structured logging
json_log_file = os.path.join(log_directory, "llm_api_calls.json")


class LLMMetadata(BaseModel):
    service_provider: str = Field(description="The service provider used (e.g., 'openai', 'groq', 'huggingface')")
    llm_model_name: str = Field(description="The name of the model used (e.g., 'gpt-4', 'llama-3.1')")
    temperature: float = Field(description="Sampling temperature used for generation")
    max_tokens: int = Field(description="Maximum number of tokens in the response")
    response_time_seconds: float = Field(description="Time taken to generate the response")
    error: Optional[str] = Field(default=None, description="Error message if the invocation failed")
    completion_tokens: Optional[int] = Field(default=None, description="Number of tokens in the completion")
    prompt_tokens: Optional[int] = Field(default=None, description="Number of tokens in the prompt")
    total_tokens: Optional[int] = Field(default=None, description="Total number of tokens used")
    estimated_cost: Optional[float] = Field(default=None, description="Estimated cost of the API call")


class LLMResponse(BaseModel):
    content: str = Field(description="Response content from the LLM")
    metadata: LLMMetadata = Field(description="Metadata associated with the response")


class LLM(BaseChatModel):
    service_provider: str = Field(description="Service provider (e.g., 'openai', 'groq', 'huggingface')")
    llm_model_name: str = Field(description="Name of the model (e.g., 'gpt-4', 'llama-3.1')")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: int = Field(default=2048, description="Maximum number of tokens in the response")
    api_key: str = Field(description='API Key of the Service Provider', default=None)
    base_url: str = Field(description='Optional Base URL for Open AI Service Provider', default=None)

    _llm: Optional[BaseChatModel] = PrivateAttr(default=None)
    _conversation_history: List[Dict[str, str]] = PrivateAttr(default_factory=list)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True
    )

    def __init__(self, **data):
        super().__init__(**data)
        self.initialize_llm()

    @property
    def _llm_type(self) -> str:
        return f"{self.service_provider}_{self.llm_model_name}"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "service_provider": self.service_provider,
            "llm_model_name": self.llm_model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

    def initialize_llm(self):
        if self.service_provider.lower() == 'groq':
            self._llm = ChatGroq(
                model=self.llm_model_name,
                temperature=self.temperature,
                api_key=""
            )
        elif self.service_provider.lower() == "openai":
            self._llm = OpenAI(
                api_key=self.api_key or os.getenv("DEEPINFRA_API_KEY"),
                base_url=self.base_url or os.getenv("BASE_URL")
            )

        elif self.service_provider.lower() == "github": 
            self._llm = ChatCompletionsClient(
                endpoint=os.getenv("endpoint"),
                credential=AzureKeyCredential(os.getenv("GITHUB_TOKEN"))
            )

        else:
            raise ValueError(f"Unsupported service provider: {self.service_provider}")

    def _generate(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> tuple:
        if not self._llm:
            raise ValueError("LLM not initialized properly.")

        if self.service_provider.lower() == "groq":
            response = self._llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            return content, response

        elif self.service_provider.lower() == "openai":
            chat_completion = self._llm.chat.completions.create(
                model=self.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )
            content = chat_completion.choices[0].message.content
            return content, chat_completion

        elif self.service_provider.lower() == "github":
        

            response = self._llm.complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=2048,
                model=self.llm_model_name,
            )

            content = response.choices[0].message.content
            return content, response

        else:
            raise ValueError(f"Unsupported service provider: {self.service_provider}")

        # This line should never be reached if all providers are handled correctly
        raise ValueError(f"No return statement executed for provider: {self.service_provider}")

    def _write_json_log(self, log_entry: Dict[str, Any]):
        """
        Write a structured log entry to the JSON log file.
        
        Args:
            log_entry: Dictionary containing log information
        """
        try:
            # Ensure the log entry has a timestamp
            if "timestamp" not in log_entry:
                log_entry["timestamp"] = datetime.datetime.now().isoformat()
                
            # Append to the JSON log file
            with open(json_log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Error writing to JSON log: {str(e)}")
    
    def invoke(self, prompt: str, **kwargs) -> LLMResponse:
        request_id = f"{int(time.time())}_{id(self)}"
        start_time = time.time()
        timestamp = datetime.datetime.now().isoformat()
        
        # Log the request
        request_log = {
            "request_id": request_id,
            "timestamp": timestamp,
            "type": "request",
            "service_provider": self.service_provider,
            "llm_model_name": self.llm_model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "prompt": prompt,
            "kwargs": str(kwargs)
        }
        
        logger.info(f"API Request: {request_id} | Provider: {self.service_provider} | Model: {self.llm_model_name}")
        
        # Write structured log to JSON file
        self._write_json_log(request_log)
        
        try:
            content, full_response = self._generate(prompt)

            end_time = time.time()
            response_time = end_time - start_time
            # Extract token usage and cost information from the response
            completion_tokens = None
            prompt_tokens = None
            total_tokens = None
            estimated_cost = None
            
            # Extract usage statistics if available
            if hasattr(full_response, 'usage'):
                usage = full_response.usage
                completion_tokens = getattr(usage, 'completion_tokens', None)
                prompt_tokens = getattr(usage, 'prompt_tokens', None)
                total_tokens = getattr(usage, 'total_tokens', None)
                estimated_cost = getattr(usage, 'estimated_cost', None)
            
            metadata = LLMMetadata(
                service_provider=self.service_provider,
                llm_model_name=self.llm_model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_time_seconds=response_time,
                error=None,
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
                estimated_cost=estimated_cost
            )
            
            # Log the successful response
            response_log = {
                "request_id": request_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "type": "response",
                "service_provider": self.service_provider,
                "llm_model_name": self.llm_model_name,
                "response_time_seconds": response_time,
                "status": "success",
                "content_length": len(content),
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
                "estimated_cost": estimated_cost,
                "response_id": getattr(full_response, 'id', None),
                "model": getattr(full_response, 'model', self.llm_model_name),
                "object": getattr(full_response, 'object', None),
                "created": getattr(full_response, 'created', None)
            }
            
            # Add full response details if possible
            try:
                if hasattr(full_response, 'model_dump'):
                    response_log["full_response"] = full_response.model_dump()
                elif hasattr(full_response, '__dict__'):
                    response_log["full_response"] = full_response.__dict__
                else:
                    response_log["full_response"] = str(full_response)
            except Exception as e:
                response_log["serialization_error"] = str(e)
            
            # Enhanced logging with token usage information
            token_info = ""
            if total_tokens:
                token_info = f" | Tokens: {total_tokens} (P:{prompt_tokens}/C:{completion_tokens})"
            cost_info = f" | Cost: ${estimated_cost:.8f}" if estimated_cost else ""
            
            logger.info(f"API Response: {request_id} | Time: {response_time:.2f}s{token_info}{cost_info} | Status: Success")
            
            # Write structured log to JSON file
            self._write_json_log(response_log)
            
            return LLMResponse(content=content, metadata=metadata)
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            error_message = str(e)
            
            # Log the error
            error_log = {
                "request_id": request_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "type": "response",
                "service_provider": self.service_provider,
                "llm_model_name": self.llm_model_name,
                "response_time_seconds": response_time,
                "status": "error",
                "error": error_message
            }
            
            logger.error(f"API Error: {request_id} | Time: {response_time:.2f}s | Error: {error_message}")
            
            # Write structured log to JSON file
            self._write_json_log(error_log)
            
            metadata = LLMMetadata(
                service_provider=self.service_provider,
                llm_model_name=self.llm_model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_time_seconds=response_time,
                error=error_message
            )
            return LLMResponse(content="", metadata=metadata)