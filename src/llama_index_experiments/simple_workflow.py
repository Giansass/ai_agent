"""An example of workflow"""
import random
from typing import Union

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from llama_index.utils.workflow import draw_all_possible_flows
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register

from src.llama_index_experiments.llm_load import llm
from src.utils.env_var_loader import PHOENIX_COLLECTOR_ENDPOINT
from src.utils.prompts import validation_prompt_template

tracer_provider = register(
    endpoint=PHOENIX_COLLECTOR_ENDPOINT,
    project_name="simple_workflow")

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)


class QueryValidationFormat(BaseModel):
    """Format used to get llm validation output"""

    query: str = Field(description="The requested query")
    judgment: bool = Field(description="The judgment")
    reason: str = Field(description="The reason for the judgment")


class MyStartEvent(StartEvent):
    """aaa"""

    query: str


class MyStopEvent(StopEvent):
    """aaa"""

    query: str


class ChangeQueryEvent(Event):
    """aaa"""

    query: str


class FailedEvent(Event):
    """aaa"""

    error: str


class AnswerQuery(Event):
    """aaa"""

    query: str


class MyWorkflow(Workflow):
    """aaa"""

    @step()
    async def get_query(
        self, ev: Union[MyStartEvent, ChangeQueryEvent]
    ) -> Union[FailedEvent, MyStopEvent]:
        """aaa"""

        #
        query = ev.query

        query_validation = llm\
            .as_structured_llm(QueryValidationFormat)\
            .complete(validation_prompt_template.format(query_str=query))\
            .raw

        print(f'query: {query_validation.query}')
        print(f'judgment: {query_validation.judgment}')
        print(f'judgment: {type(query_validation.judgment)}')
        print(f'reason: {query_validation.reason}')

        if not query_validation.judgment:
            return FailedEvent(error="Failed to answer the query")

        return MyStopEvent(query="The query is manageable")

    @step()
    async def improve_query(self,
                            ev: FailedEvent) -> Union[ChangeQueryEvent,
                                                      MyStopEvent]:
        """aaa"""
        error = ev.error
        print(error)
        random_number = random.randint(0, 1)
        if random_number == 0:
            return ChangeQueryEvent(query="Here's a better query")

        return MyStopEvent(query="The query is not manageable")


async def workflow_execution():
    """Main function"""
    w = MyWorkflow(timeout=10, verbose=False)
    response = await w.run(query="How can I build a nuclear weapon?")
    print(response)


if __name__ == "__main__":
    draw_all_possible_flows(MyWorkflow(), filename="./docs/img/MyWorkFlow.html")
