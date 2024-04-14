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
    JSON = "json"


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
    UKRAINIAN = "uk"
    URDU = "ur"
    UZBEK = "uz"
    VIETNAMESE = "vi"


SUPPORTED_FILE_TYPES = [
    ".pdf",
    # Microsoft word - all versions
    ".doc",
    ".docx",
    ".docm",
    ".dot",
    ".dotx",
    ".dotm",
    # Rich text format
    ".rtf",
    # Microsoft Works
    ".wps",
    # Word Perfect
    ".wpd",
    # Open Office
    ".sxw",
    ".stw",
    ".sxg",
    # Apple
    ".pages",
    # Mac Write
    ".mw",
    ".mcw",
    # Unified Office Format text
    ".uot",
    ".uof",
    ".uos",
    ".uop",
    # Microsoft powerpoints
    ".ppt",
    ".pptx",
    ".pot",
    ".pptm",
    ".potx",
    ".potm",
    # Apple keynote
    ".key",
    # Open Office Presentations
    ".odp",
    ".odg",
    ".otp",
    ".fopd",
    ".sxi",
    ".sti",
    # ebook
    ".epub",
]


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
        description="The number of workers to use sending API requests for parsing.",
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
    show_progress: bool = Field(
        default=True, description="Show progress when parsing multiple files."
    )
    language: Language = Field(
        default=Language.ENGLISH, description="The language of the text to parse."
    )
    parsing_instruction: Optional[str] = Field(
        default="", description="The parsing instruction for the parser."
    )
    ignore_errors: bool = Field(
        default=True,
        description="Whether or not to ignore and skip errors raised during parsing.",
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

    # upload a document and get back a job_id
    async def _create_job(
        self, file_path: str, extra_info: Optional[dict] = None
    ) -> str:
        file_path = str(file_path)
        file_ext = os.path.splitext(file_path)[1]
        if file_ext not in SUPPORTED_FILE_TYPES:
            raise Exception(
                f"Currently, only the following file types are supported: {SUPPORTED_FILE_TYPES}\n"
                f"Current file type: {file_ext}"
            )

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
                response = await client.post(
                    url,
                    files=files,
                    headers=headers,
                    data={
                        "language": self.language.value,
                        "parsing_instruction": self.parsing_instruction,
                    },
                )
                if not response.is_success:
                    raise Exception(f"Failed to parse the file: {response.text}")

        # check the status of the job, return when done
        job_id = response.json()["id"]
        return job_id

    async def _get_job_result(
        self, job_id: str, result_type: str, verbose: bool = False
    ) -> dict:
        result_url = f"{self.base_url}/api/parsing/job/{job_id}/result/{result_type}"
        status_url = f"{self.base_url}/api/parsing/job/{job_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        start = time.time()
        tries = 0
        while True:
            await asyncio.sleep(self.check_interval)
            async with httpx.AsyncClient(timeout=self.max_timeout) as client:
                tries += 1

                result = await client.get(status_url, headers=headers)

                if result.status_code != 200:
                    end = time.time()
                    if end - start > self.max_timeout:
                        raise Exception(f"Timeout while parsing the file: {job_id}")
                    if verbose and tries % 10 == 0:
                        print(".", end="", flush=True)

                    await asyncio.sleep(self.check_interval)

                    continue

                # Allowed values "PENDING", "SUCCESS", "ERROR", "CANCELED"
                status = result.json()["status"]
                if status == "SUCCESS":
                    parsed_result = await client.get(result_url, headers=headers)
                    return parsed_result.json()
                elif status == "PENDING":
                    end = time.time()
                    if end - start > self.max_timeout:
                        raise Exception(f"Timeout while parsing the file: {job_id}")
                    if verbose and tries % 10 == 0:
                        print(".", end="", flush=True)

                    await asyncio.sleep(self.check_interval)

                    continue
                else:
                    raise Exception(
                        f"Failed to parse the file: {job_id}, status: {status}"
                    )

    async def _aload_data(
        self, file_path: str, extra_info: Optional[dict] = None, verbose: bool = False
    ) -> List[Document]:
        """Load data from the input path."""
        try:
            job_id = await self._create_job(file_path, extra_info=extra_info)
            if verbose:
                print("Started parsing the file under job_id %s" % job_id)

            result = await self._get_job_result(
                job_id, self.result_type.value, verbose=verbose
            )

            return [
                Document(
                    text=result[self.result_type.value],
                    metadata=extra_info or {},
                )
            ]

        except Exception as e:
            print(f"Error while parsing the file '{file_path}':", e)
            if self.ignore_errors:
                return []
            else:
                raise e

    async def aload_data(
        self, file_path: Union[List[str], str], extra_info: Optional[dict] = None
    ) -> List[Document]:
        """Load data from the input path."""
        if isinstance(file_path, (str, Path)):
            return await self._aload_data(
                file_path, extra_info=extra_info, verbose=self.verbose
            )
        elif isinstance(file_path, list):
            jobs = [
                self._aload_data(
                    f,
                    extra_info=extra_info,
                    verbose=self.verbose and not self.show_progress,
                )
                for f in file_path
            ]
            try:
                results = await run_jobs(
                    jobs,
                    workers=self.num_workers,
                    desc="Parsing files",
                    show_progress=self.show_progress,
                )

                # return flattened results
                return [item for sublist in results for item in sublist]
            except RuntimeError as e:
                if nest_asyncio_err in str(e):
                    raise RuntimeError(nest_asyncio_msg)
                else:
                    raise e
        else:
            raise ValueError(
                "The input file_path must be a string or a list of strings."
            )

    def load_data(
        self, file_path: Union[List[str], str], extra_info: Optional[dict] = None
    ) -> List[Document]:
        """Load data from the input path."""
        try:
            return asyncio.run(self.aload_data(file_path, extra_info))
        except RuntimeError as e:
            if nest_asyncio_err in str(e):
                raise RuntimeError(nest_asyncio_msg)
            else:
                raise e

    async def _aget_json(
        self, file_path: str, extra_info: Optional[dict] = None
    ) -> List[dict]:
        """Load data from the input path."""
        try:
            job_id = await self._create_job(file_path, extra_info=extra_info)
            if self.verbose:
                print("Started parsing the file under job_id %s" % job_id)

            result = await self._get_job_result(job_id, "json")
            result["job_id"] = job_id
            result["file_path"] = file_path
            return [result]

        except Exception as e:
            print(f"Error while parsing the file '{file_path}':", e)
            if self.ignore_errors:
                return []
            else:
                raise e

    async def aget_json(
        self, file_path: Union[List[str], str], extra_info: Optional[dict] = None
    ) -> List[dict]:
        """Load data from the input path."""
        if isinstance(file_path, (str, Path)):
            return await self._aget_json(file_path, extra_info=extra_info)
        elif isinstance(file_path, list):
            jobs = [self._aget_json(f, extra_info=extra_info) for f in file_path]
            try:
                results = await run_jobs(
                    jobs,
                    workers=self.num_workers,
                    desc="Parsing files",
                    show_progress=self.show_progress,
                )

                # return flattened results
                return [item for sublist in results for item in sublist]
            except RuntimeError as e:
                if nest_asyncio_err in str(e):
                    raise RuntimeError(nest_asyncio_msg)
                else:
                    raise e
        else:
            raise ValueError(
                "The input file_path must be a string or a list of strings."
            )

    def get_json_result(
        self, file_path: Union[List[str], str], extra_info: Optional[dict] = None
    ) -> List[dict]:
        """Parse the input path."""
        try:
            return asyncio.run(self.aget_json(file_path, extra_info))
        except RuntimeError as e:
            if nest_asyncio_err in str(e):
                raise RuntimeError(nest_asyncio_msg)
            else:
                raise e

    def get_images(self, json_result: List[dict], download_path: str) -> List[dict]:
        """Download images from the parsed result."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            images = []
            for result in json_result:
                job_id = result["job_id"]
                for page in result["pages"]:
                    if self.verbose:
                        print(f"> Image for page {page['page']}: {page['images']}")
                    for image in page["images"]:
                        image_name = image["name"]
                        image_path = os.path.join(
                            download_path, f"{job_id}-{image_name}"
                        )
                        image["path"] = image_path
                        image["job_id"] = job_id
                        image["original_pdf_path"] = result["file_path"]
                        image["page_number"] = page["page"]
                        with open(image_path, "wb") as f:
                            image_url = f"{self.base_url}/api/parsing/job/{job_id}/result/image/{image_name}"
                            f.write(httpx.get(image_url, headers=headers).content)
                        images.append(image)
            return images
        except Exception as e:
            print("Error while downloading images from the parsed result:", e)
            if self.ignore_errors:
                return []
            else:
                raise e
