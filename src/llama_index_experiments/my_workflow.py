"""An example of workflow"""
import random
from typing import Union

from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step


class FailedEvent(Event):
    """aaa"""

    error: str


class QueryEvent(Event):
    """aaa"""

    query: str


class MyWorkflow(Workflow):
    """aaa"""

    @step()
    async def answer_query(
        self, ev: Union[StartEvent, QueryEvent]
    ) -> Union[FailedEvent, StopEvent]:
        """aaa"""
        query = ev.query
        print(query)
        random_number = random.randint(0, 1)
        if random_number == 0:
            return FailedEvent(error="Failed to answer the query")

        return StopEvent(result="I can answer your query")

    @step()
    async def improve_query(self, ev: FailedEvent) -> Union[QueryEvent, StopEvent]:
        """aaa"""
        error = ev.error
        print(error)
        random_number = random.randint(0, 1)
        if random_number == 0:
            return QueryEvent(query="Here's a better query")

        return StopEvent(result="Your query cannot be fixed")


async def workflow_execution():
    """Main function"""
    w = MyWorkflow(timeout=10, verbose=False)
    response = await w.run(query="Hi")
    print(response)


if __name__ == "__main__":
    from llama_index.utils.workflow import draw_all_possible_flows

    draw_all_possible_flows(MyWorkflow(), filename="./docs/img/MyWorkFlow.html")
