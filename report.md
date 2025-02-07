# LlamaReport (beta/invite-only)

LlamaReport is a prebuilt agentic report builder that can be used to build reports from a variety of data sources.

The python SDK for interacting with the LlamaReport API. The SDK provides two main classes:

- `LlamaReport`: For managing reports (create, list, delete)
- `ReportClient`: For working with a specific report (editing, approving, etc.)

## Quickstart

```bash
pip install llama-cloud-services
```

```python
from llama_cloud_services import LlamaReport

# Initialize the client
client = LlamaReport(
    api_key="your-api-key",
    # Optional: Specify project_id, organization_id, async_httpx_client
)

# Create a new report
report = client.create_report(
    "My Report",
    # must have one of template_text or template_instructions
    template_text="Your template text",
    template_instructions="Instructions for the template",
    # must have one of input_files or retriever_id
    input_files=["data1.pdf", "data2.pdf"],
    retriever_id="retriever-id",
)
```

## Working with Reports

The typical workflow for a report involves:

1. Creating the report
2. Waiting for and approving the plan
3. Waiting for report generation
4. Making edits to the report

Here's a complete example:

```python
# Create a report
report = client.create_report(
    "Quarterly Analysis", input_files=["q1_data.pdf", "q2_data.pdf"]
)

# Wait for the plan to be ready
plan = report.wait_for_plan()

# Option 1: Directly approve the plan
report.update_plan(action="approve")

# Option 2: Suggest and review edits to the plan
suggestions = report.suggest_edits(
    "Can you add a section about market trends?"
)
for suggestion in suggestions:
    print(suggestion)

    # Accept or reject the suggestion
    if input("Accept? (y/n): ").lower() == "y":
        report.accept_edit(suggestion)
    else:
        report.reject_edit(suggestion)

# Wait for the report to complete
report = report.wait_for_completion()

# Make edits to the final report
suggestions = report.suggest_edits("Make the executive summary more concise")

# Review and accept/reject suggestions as above
...
```

### Getting the Final Report

Once you are satisfied with the report, you can get the final report object and use the content as you see fit.

Here's an example of printing out the final report:

```python
report = report.get()
report_text = "\n\n".join([block.template for block in report.blocks])

print(report_text)
```

## Additional Features

- **Async Support**: All methods have async counterparts: `create_report` -> `acreate_report`, `wait_for_plan` -> `await_for_plan`, etc.
- **Automatic Chat History**: The SDK automatically keeps track of chat history for each suggestion, unless you specify `auto_history=False` in `suggest_edits`.
- **Custom HTTP Client**: You can provide your own `httpx.AsyncClient` to the `LlamaReport` class.
- **Project and Organization IDs**: You can specify `project_id` and `organization_id` to use a specific project or organization.
