import os
import asyncio
from io import TextIOWrapper

import httpx
import mimetypes
import time
from pathlib import Path, PurePath, PurePosixPath
from typing import AsyncGenerator, Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager
from io import BufferedIOBase

from fsspec import AbstractFileSystem
from fsspec.spec import AbstractBufferedFile
from llama_index.core.async_utils import run_jobs
from llama_index.core.bridge.pydantic import Field, field_validator
from llama_index.core.constants import DEFAULT_BASE_URL
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.readers.file.base import get_default_fs
from llama_index.core.schema import Document
from llama_parse.utils import (
    nest_asyncio_err,
    nest_asyncio_msg,
    ResultType,
    Language,
    SUPPORTED_FILE_TYPES,
)
from copy import deepcopy

# can put in a path to the file or the file bytes itself
# if passing as bytes or a buffer, must provide the file_name in extra_info
FileInput = Union[str, bytes, BufferedIOBase]

_DEFAULT_SEPARATOR = "\n---\n"


class LlamaParseParams(BasePydanticReader):
    bounding_box: Optional[str] = Field(
        default=None,
        description="The bounding box to use to extract text from documents describe as a string containing the bounding box margins",
    )
   
    continuous_mode: Optional[bool] = Field(
        default=False,
        description="If set to true, the parser will to merge together following tables",
    )
    do_not_cache: Optional[bool] = Field(
        default=False,
        description="If set to true, the document will not be cached. This mean that you will be re-charged it you reprocess them as they will not be cached.",
    )
    do_not_unroll_columns: Optional[bool] = Field(
        default=False,
        description="If set to true, the parser will keep column in the text according to document layout. Reduce reconstruction accuracy, and LLM's/embedings performances in most case.",
    )
    fast_mode: Optional[bool] = Field(
        default=False,
        description="Note: Non compatible with other modes. If set to true, the parser will use a faster mode to extract text from documents. This mode will skip OCR of images, and table/heading reconstruction.",
    )
    gpt4o_api_key: Optional[str] = Field(
        default=None,
        description="(deprecated use vendor_multimodal_model_name='gpt-4o' instead). The API key for the GPT-4o API. Lowers the cost of parsing.",
    )
    gpt4o_mode: bool = Field(
        default=False,
        description="(deprecated use vendor_multimodal_api_key='gpt-4o' instead). Whether to use gpt-4o extract text from documents.",
    )
    guess_xlsx_sheet_name: Optional[str] = Field(
        default=False,
        description="Experimental: If set to true, when outputting to xlsx, the parser will try to guess the sheet name based on the context of the table.",
    )
    ignore_errors: bool = Field(
        default=True,
        description="Whether or not to ignore and skip errors raised during parsing.",
    )
    invalidate_cache: Optional[bool] = Field(
        default=False,
        description="If set to true, the cache will be ignored and the document re-processes. All document are kept in cache for 48hours after the job was completed to avoid processing the same document twice.",
    )
    language: Language = Field(
        default=Language.ENGLISH, description="The language of the text to parse."
    )
    page_prefix: Optional[str] = Field(
        default=None,
        description="A templated prefix to add to the beginning of each page. If it contain `{page_number}`, it will be replaced by the page number.",
    )
    page_separator: Optional[str] = Field(
        default=None,
        description="A templated  page separator to use to split the text.  If it contain `{page_number}`,it will be replaced by the next page number. If not set will the default separator '\\n---\\n' will be used.",
    )
    page_suffix: Optional[str] = Field(
        default=None,
        description="A templated suffix to add to the beginning of each page. If it contain `{page_number}`, it will be replaced by the page number.",
    )
    parsing_instruction: Optional[str] = Field(
        default="", description="The parsing instruction for the parser."
    )
    skip_diagonal_text: Optional[bool] = Field(
        default=False,
        description="If set to true, the parser will ignore diagonal text (when the text rotation in degrees modulo 90 is not 0).",
    )
    split_by_page: bool = Field(
        default=True,
        description="Whether to split by page using the page separator",
    )
    take_screenshot: bool = Field(
        default=False,
        description="Whether to take screenshot of each page of the document.",
    )
    target_pages: Optional[str] = Field(
        default=None,
        description="The target pages to extract text from documents. Describe as a comma separated list of page numbers. The first page of the document is page 0",
    )
    use_vendor_multimodal_model: bool = Field(
        default=False,
        description="Whether to use the vendor multimodal API.",
    )
    vendor_multimodal_api_key: Optional[str] = Field(
        default=None,
        description="The API key for the multimodal API.",
    )
   
    vendor_multimodal_model_name: Optional[str] = Field(
        default=None,
        description="The model name for the vendor multimodal API.",
    )

class LlamaParse(LlamaParseParams):
    """A smart-parser for files."""

    """Package parameters"""
    api_key: str = Field(
        default="",
        description="The API key for the LlamaParse API.",
        validate_default=True,
    )
    base_url: str = Field(
        default=DEFAULT_BASE_URL,
        description="The base URL of the Llama Parsing API.",
    )
    check_interval: int = Field(
        default=1,
        description="The interval in seconds to check if the parsing is done.",
    )
    custom_client: Optional[httpx.AsyncClient] = Field(
        default=None, description="A custom HTTPX client to use for sending requests."
    )
    num_workers: int = Field(
        default=4,
        gt=0,
        lt=10,
        description="The number of workers to use sending API requests for parsing.",
    )
    max_timeout: int = Field(
        default=2000,
        description="The maximum timeout in seconds to wait for the parsing to finish.",
    )
    result_type: ResultType = Field(
        default=ResultType.TXT, description="The result type for the parser."
    )
    show_progress: bool = Field(
        default=True, description="Show progress when parsing multiple files."
    )
    verbose: bool = Field(
        default=True, description="Whether to print the progress of the parsing."
    )
   
    @field_validator("api_key", mode="before", check_fields=True)
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate the API key."""
        if not v:
            import os

            api_key = os.getenv("LLAMA_CLOUD_API_KEY", None)
            if api_key is None:
                raise ValueError("The API key is required.")
            return api_key

        return v

    @field_validator("base_url", mode="before", check_fields=True)
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """Validate the base URL."""
        url = os.getenv("LLAMA_CLOUD_BASE_URL", None)
        return url or v or DEFAULT_BASE_URL

    @asynccontextmanager
    async def client_context(self) -> AsyncGenerator[httpx.AsyncClient, None]:
        """Create a context for the HTTPX client."""
        if self.custom_client is not None:
            yield self.custom_client
        else:
            async with httpx.AsyncClient(timeout=self.max_timeout) as client:
                yield client

    # upload a document and get back a job_id
    async def _create_job(
        self,
        file_input: FileInput,
        extra_info: Optional[dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/api/parsing/upload"
        files = None
        file_handle = None

        if isinstance(file_input, (bytes, BufferedIOBase)):
            if not extra_info or "file_name" not in extra_info:
                raise ValueError(
                    "file_name must be provided in extra_info when passing bytes"
                )
            file_name = extra_info["file_name"]
            mime_type = mimetypes.guess_type(file_name)[0]
            files = {"file": (file_name, file_input, mime_type)}
        elif isinstance(file_input, (str, Path, PurePosixPath, PurePath)):
            file_path = str(file_input)
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in SUPPORTED_FILE_TYPES:
                raise Exception(
                    f"Currently, only the following file types are supported: {SUPPORTED_FILE_TYPES}\n"
                    f"Current file type: {file_ext}"
                )
            mime_type = mimetypes.guess_type(file_path)[0]
            # Open the file here for the duration of the async context
            # load data, set the mime type
            fs = fs or get_default_fs()
            file_handle = fs.open(file_input, "rb")
            files = {"file": (os.path.basename(file_path), file_handle, mime_type)}
        else:
            raise ValueError(
                "file_input must be either a file path string, file bytes, or buffer object"
            )

        data = {
        }   

        # for each key of LlamaParseParams
        # if the value is not None, add it to the data
        for key in LlamaParseParams:
            if getattr(self, key) is not None:
                data[key] = getattr(self, key)

        try:
            async with self.client_context() as client:
                response = await client.post(
                    url,
                    files=files,
                    headers=headers,
                    data=data,
                )
                if not response.is_success:
                    raise Exception(f"Failed to parse the file: {response.text}")
                job_id = response.json()["id"]
                return job_id
        finally:
            if file_handle is not None:
                file_handle.close()

    @staticmethod
    def __get_filename(f: Union[TextIOWrapper, AbstractBufferedFile]) -> str:
        if isinstance(f, TextIOWrapper):
            return f.name
        return f.full_name

    async def _get_job_result(
        self, job_id: str, result_type: str, verbose: bool = False
    ) -> Dict[str, Any]:
        result_url = f"{self.base_url}/api/parsing/job/{job_id}/result/{result_type}"
        status_url = f"{self.base_url}/api/parsing/job/{job_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        start = time.time()
        tries = 0
        while True:
            await asyncio.sleep(self.check_interval)
            async with self.client_context() as client:
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

    async def _aload_data(
        self,
        file_path: FileInput,
        extra_info: Optional[dict] = None,
        fs: Optional[AbstractFileSystem] = None,
        verbose: bool = False,
    ) -> List[Document]:
        """Load data from the input path."""
        try:
            job_id = await self._create_job(file_path, extra_info=extra_info, fs=fs)
            if verbose:
                print("Started parsing the file under job_id %s" % job_id)

            result = await self._get_job_result(
                job_id, self.result_type.value, verbose=verbose
            )

            docs = [
                Document(
                    text=result[self.result_type.value],
                    metadata=extra_info or {},
                )
            ]
            if self.split_by_page:
                return self._get_sub_docs(docs)
            else:
                return docs

        except Exception as e:
            file_repr = file_path if isinstance(file_path, str) else "<bytes/buffer>"
            print(f"Error while parsing the file '{file_repr}':", e)
            if self.ignore_errors:
                return []
            else:
                raise e

    async def aload_data(
        self,
        file_path: Union[List[FileInput], FileInput],
        extra_info: Optional[dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Load data from the input path."""
        if isinstance(file_path, (str, Path, bytes, BufferedIOBase)):
            return await self._aload_data(
                file_path, extra_info=extra_info, fs=fs, verbose=self.verbose
            )
        elif isinstance(file_path, list):
            jobs = [
                self._aload_data(
                    f,
                    extra_info=extra_info,
                    fs=fs,
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
        self,
        file_path: Union[List[FileInput], FileInput],
        extra_info: Optional[dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Load data from the input path."""
        try:
            return asyncio.run(self.aload_data(file_path, extra_info, fs=fs))
        except RuntimeError as e:
            if nest_asyncio_err in str(e):
                raise RuntimeError(nest_asyncio_msg)
            else:
                raise e

    async def _aget_json(
        self, file_path: FileInput, extra_info: Optional[dict] = None
    ) -> List[dict]:
        """Load data from the input path."""
        try:
            job_id = await self._create_job(file_path, extra_info=extra_info)
            if self.verbose:
                print("Started parsing the file under job_id %s" % job_id)
            result = await self._get_job_result(job_id, "json")
            result["job_id"] = job_id

            if not isinstance(file_path, (bytes, BufferedIOBase)):
                result["file_path"] = str(file_path)

            return [result]
        except Exception as e:
            file_repr = file_path if isinstance(file_path, str) else "<bytes/buffer>"
            print(f"Error while parsing the file '{file_repr}':", e)
            if self.ignore_errors:
                return []
            else:
                raise e

    async def aget_json(
        self,
        file_path: Union[List[FileInput], FileInput],
        extra_info: Optional[dict] = None,
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
        self,
        file_path: Union[List[FileInput], FileInput],
        extra_info: Optional[dict] = None,
    ) -> List[dict]:
        """Parse the input path."""
        try:
            return asyncio.run(self.aget_json(file_path, extra_info))
        except RuntimeError as e:
            if nest_asyncio_err in str(e):
                raise RuntimeError(nest_asyncio_msg)
            else:
                raise e

    async def aget_images(
        self, json_result: List[dict], download_path: str
    ) -> List[dict]:
        """Download images from the parsed result."""
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # make the download path
        if not os.path.exists(download_path):
            os.makedirs(download_path)

        try:
            images = []
            for result in json_result:
                job_id = result["job_id"]
                for page in result["pages"]:
                    if self.verbose:
                        print(f"> Image for page {page['page']}: {page['images']}")
                    for image in page["images"]:
                        image_name = image["name"]

                        # get the full path
                        image_path = os.path.join(
                            download_path, f"{job_id}-{image_name}"
                        )

                        # get a valid image path
                        if not image_path.endswith(".png"):
                            if not image_path.endswith(".jpg"):
                                image_path += ".png"

                        image["path"] = image_path
                        image["job_id"] = job_id

                        image["original_file_path"] = result.get("file_path", None)

                        image["page_number"] = page["page"]
                        with open(image_path, "wb") as f:
                            image_url = f"{self.base_url}/api/parsing/job/{job_id}/result/image/{image_name}"
                            async with self.client_context() as client:
                                res = await client.get(
                                    image_url, headers=headers, timeout=self.max_timeout
                                )
                                res.raise_for_status()
                                f.write(res.content)
                        images.append(image)
            return images
        except Exception as e:
            print("Error while downloading images from the parsed result:", e)
            if self.ignore_errors:
                return []
            else:
                raise e

    def get_images(self, json_result: List[dict], download_path: str) -> List[dict]:
        """Download images from the parsed result."""
        try:
            return asyncio.run(self.aget_images(json_result, download_path))
        except RuntimeError as e:
            if nest_asyncio_err in str(e):
                raise RuntimeError(nest_asyncio_msg)
            else:
                raise e

    def _get_sub_docs(self, docs: List[Document]) -> List[Document]:
        """Split docs into pages, by separator."""
        sub_docs = []
        separator = self.page_separator or _DEFAULT_SEPARATOR
        for doc in docs:
            doc_chunks = doc.text.split(separator)
            for doc_chunk in doc_chunks:
                sub_doc = Document(
                    text=doc_chunk,
                    metadata=deepcopy(doc.metadata),
                )
                sub_docs.append(sub_doc)

        return sub_docs
