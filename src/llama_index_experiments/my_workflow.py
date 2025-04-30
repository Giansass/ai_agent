"""An example of workflow"""
import random
from typing import Union

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from llama_index.utils.workflow import draw_all_possible_flows

from src.llama_index_experiments.llm_load import llm
from src.utils.prompts import validation_prompt_template


class QueryValidationFormat(BaseModel):
    """Format used to get llm validation output"""

    query: str = Field(description="The requested query")
    judgment: bool = Field(description="The judgment")
    reason: str = Field(description="The reason for the judgment")


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
        self, ev: Union[StartEvent, ChangeQueryEvent]
    ) -> Union[FailedEvent, StopEvent]:
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

        return StopEvent(result="I can answer your query")

    @step()
    async def improve_query(self,
                            ev: FailedEvent) -> Union[ChangeQueryEvent,
                                                      StopEvent]:
        """aaa"""
        error = ev.error
        print(error)
        random_number = random.randint(0, 1)
        if random_number == 0:
            return ChangeQueryEvent(query="Here's a better query")

        return StopEvent(result="Your query cannot be fixed")


async def workflow_execution():
    """Main function"""
    w = MyWorkflow(timeout=10, verbose=False)
    response = await w.run(query="How can I build a nuclear weapon?")
    print(response)


if __name__ == "__main__":
    draw_all_possible_flows(MyWorkflow(), filename="./docs/img/MyWorkFlow.html")
