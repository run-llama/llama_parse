import os
import asyncio
import httpx
import mimetypes
import time
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

from llama_index.core.async_utils import run_jobs
from llama_index.core.bridge.pydantic import Field, validator
from llama_index.core.constants import DEFAULT_BASE_URL
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

nest_asyncio_err = "cannot be called from a running event loop"
nest_asyncio_msg = "The event loop is already running. Add `import nest_asyncio; nest_asyncio.apply()` to your code to fix this issue."

class ResultType(str, Enum):
    """The result type for the parser."""

    TXT = "text"
    MD = "markdown"

class Language(str, Enum):
    BAZA = "abq"
    ADYGHE = "ady"
    AFRIKAANS = "af"
    ANGIKA = "ang"
    ARABIC = "ar"
    ASSAMESE = "as"
    AVAR = "ava"
    AZERBAIJANI = "az"
    BELARUSIAN = "be"
    BULGARIAN = "bg"
    BIHARI = "bh"
    BHOJPURI = "bho"
    BENGALI = "bn"
    BOSNIAN = "bs"
    SIMPLIFIED_CHINESE = "ch_sim"
    TRADITIONAL_CHINESE = "ch_tra"
    CHECHEN = "che"
    CZECH = "cs"
    WELSH = "cy"
    DANISH = "da"
    DARGWA = "dar"
    GERMAN = "de"
    ENGLISH = "en"
    SPANISH = "es"
    ESTONIAN = "et"
    PERSIAN_FARSI = "fa"
    FRENCH = "fr"
    IRISH = "ga"
    GOAN_KONKANI = "gom"
    HINDI = "hi"
    CROATIAN = "hr"
    HUNGARIAN = "hu"
    INDONESIAN = "id"
    INGUSH = "inh"
    ICELANDIC = "is"
    ITALIAN = "it"
    JAPANESE = "ja"
    KABARDIAN = "kbd"
    KANNADA = "kn"
    KOREAN = "ko"
    KURDISH = "ku"
    LATIN = "la"
    LAK = "lbe"
    LEZGHIAN = "lez"
    LITHUANIAN = "lt"
    LATVIAN = "lv"
    MAGAHI = "mah"
    MAITHILI = "mai"
    MAORI = "mi"
    MONGOLIAN = "mn"
    MARATHI = "mr"
    MALAY = "ms"
    MALTESE = "mt"
    NEPALI = "ne"
    NEWARI = "new"
    DUTCH = "nl"
    NORWEGIAN = "no"
    OCCITAN = "oc"
    PALI = "pi"
    POLISH = "pl"
    PORTUGUESE = "pt"
    ROMANIAN = "ro"
    RUSSIAN = "ru"
    SERBIAN_CYRILLIC = "rs_cyrillic"
    SERBIAN_LATIN = "rs_latin"
    NAGPURI = "sck"
    SLOVAK = "sk"
    SLOVENIAN = "sl"
    ALBANIAN = "sq"
    SWEDISH = "sv"
    SWAHILI = "sw"
    TAMIL = "ta"
    TABASSARAN = "tab"
    TELUGU = "te"
    THAI = "th"
    TAJIK = "tjk"
    TAGALOG = "tl"
    TURKISH = "tr"
    UYGHUR = "ug"
    UKRANIAN = "uk"
    URDU = "ur"
    UZBEK = "uz"
    VIETNAMESE = "vi"


class LlamaParse(BasePydanticReader):
    """A smart-parser for files."""

    api_key: str = Field(default="", description="The API key for the LlamaParse API.")
    base_url: str = Field(
        default=DEFAULT_BASE_URL,
        description="The base URL of the Llama Parsing API.",
    )
    result_type: ResultType = Field(
        default=ResultType.TXT, description="The result type for the parser."
    )
    num_workers: int = Field(
        default=4,
        gt=0,
        lt=10, 
        description="The number of workers to use sending API requests for parsing."
    )
    check_interval: int = Field(
        default=1,
        description="The interval in seconds to check if the parsing is done.",
    )
    max_timeout: int = Field(
        default=2000,
        description="The maximum timeout in seconds to wait for the parsing to finish.",
    )
    verbose: bool = Field(
        default=True, description="Whether to print the progress of the parsing."
    )
    language: Optional[str] = Field(
         default=Language.ENGLISH, description="The language of the text to parse."
    )

    @validator("api_key", pre=True, always=True)
    def validate_api_key(cls, v: str) -> str:
        """Validate the API key."""
        if not v:
            import os
            api_key = os.getenv("LLAMA_CLOUD_API_KEY", None)
            if api_key is None:
                raise ValueError("The API key is required.")
            return api_key
        
        return v
    
    @validator("base_url", pre=True, always=True)
    def validate_base_url(cls, v: str) -> str:
        """Validate the base URL."""
        url = os.getenv("LLAMA_CLOUD_BASE_URL", None)
        return url or v or DEFAULT_BASE_URL

    async def _aload_data(self, file_path: str, extra_info: Optional[dict] = None) -> List[Document]:
        """Load data from the input path."""
        try:
            file_path = str(file_path)
            if not file_path.endswith(".pdf"):
                raise Exception("Currently, only PDF files are supported.")

            extra_info = extra_info or {}
            extra_info["file_path"] = file_path

            headers = {"Authorization": f"Bearer {self.api_key}"}

            # load data, set the mime type
            with open(file_path, "rb") as f:
                mime_type = mimetypes.guess_type(file_path)[0]
                files = {"file": (f.name, f, mime_type)}

                # send the request, start job
                url = f"{self.base_url}/api/parsing/upload"
                async with httpx.AsyncClient(timeout=self.max_timeout) as client:
                    response = await client.post(url, files=files, headers=headers, data={"language": self.language})
                    if not response.is_success:
                        raise Exception(f"Failed to parse the PDF file: {response.text}")

            # check the status of the job, return when done
            job_id = response.json()["id"]
            if self.verbose:
                print("Started parsing the file under job_id %s" % job_id)
            
            result_url = f"{self.base_url}/api/parsing/job/{job_id}/result/{self.result_type.value}"

            start = time.time()
            tries = 0
            while True:
                await asyncio.sleep(self.check_interval)
                async with httpx.AsyncClient(timeout=self.max_timeout) as client: 
                    tries += 1   
                    
                    result = await client.get(result_url, headers=headers)

                    if result.status_code == 404:
                        end = time.time()
                        if end - start > self.max_timeout:
                            raise Exception(
                                f"Timeout while parsing the PDF file: {response.text}"
                            )
                        if self.verbose and tries % 10 == 0:
                            print(".", end="", flush=True)
                        continue

                    if result.status_code == 400:
                        detail = result.json().get("detail", "Unknown error")
                        raise Exception(f"Failed to parse the PDF file: {detail}")

                    return [
                        Document(
                            text=result.json()[self.result_type.value],
                            metadata=extra_info,
                        )
                    ]
        except Exception as e:
            print("Error while parsing the PDF file: ", e)
            return []
    
    async def aload_data(self, file_path: Union[List[str], str], extra_info: Optional[dict] = None) -> List[Document]:
        """Load data from the input path."""
        if isinstance(file_path, (str, Path)):
            return await self._aload_data(file_path, extra_info=extra_info)
        elif isinstance(file_path, list):
            jobs = [self._aload_data(f, extra_info=extra_info) for f in file_path]
            try:
                results = await run_jobs(jobs, workers=self.num_workers)
                
                # return flattened results
                return [item for sublist in results for item in sublist]
            except RuntimeError as e:
                if nest_asyncio_err in str(e):
                    raise RuntimeError(nest_asyncio_msg)
                else:
                    raise e
        else:
            raise ValueError("The input file_path must be a string or a list of strings.")

    def load_data(self, file_path: Union[List[str], str], extra_info: Optional[dict] = None) -> List[Document]:
        """Load data from the input path."""
        try:
            return asyncio.run(self.aload_data(file_path, extra_info))
        except RuntimeError as e:
            if nest_asyncio_err in str(e):
                raise RuntimeError(nest_asyncio_msg)
            else:
                raise e
