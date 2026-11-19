"""The rebooking assistant as a LangGraph graph: find the fare rules, check seats, write the answer."""

import random
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from assistant import booking, llm
from assistant.data import PASSAGES
from assistant.retrieval import CHOSEN_RETRIEVER, RETRIEVERS


class State(TypedDict, total=False):
    question: str
    flight: str
    passages: list
    sources: list
    seats: int | None
    booking_failed: bool
    attempts: int
    answer: str


MAX_BOOKING_ATTEMPTS = 3
BACKOFF_BASE_SECONDS = 0.2   # wait before the 2nd attempt; doubles each retry
BACKOFF_MAX_SECONDS = 2.0    # cap on a single wait
BACKOFF_JITTER = 0.5         # +/- 50% random spread so many clients don't retry in lockstep


def backoff_delay(retry_number):
    """Seconds to wait before retry `retry_number` (1 = first retry): exponential, capped, with jitter."""
    delay = min(BACKOFF_MAX_SECONDS, BACKOFF_BASE_SECONDS * 2 ** (retry_number - 1))
    return delay * random.uniform(1 - BACKOFF_JITTER, 1 + BACKOFF_JITTER)


def find_rules(state):
    passages = RETRIEVERS[CHOSEN_RETRIEVER](state["question"], PASSAGES, 3)
    return {"passages": passages, "sources": [p["id"] for p in passages]}


def check_seats(state):
    previous = state.get("attempts", 0)
    if previous:
        time.sleep(backoff_delay(previous))
    attempts = previous + 1
    try:
        return {"seats": booking.free_seats(state["flight"]), "booking_failed": False, "attempts": attempts}
    except booking.BookingSystemDown:
        return {"seats": None, "booking_failed": True, "attempts": attempts}


def after_check_seats(state):
    # Retry while the booking system is down, up to MAX_BOOKING_ATTEMPTS; then answer without seats.
    if state["booking_failed"] and state["attempts"] < MAX_BOOKING_ATTEMPTS:
        return "check_seats"
    return "write_answer"


def write_answer(state):
    return {"answer": llm.write_answer(state["question"], state["passages"], state["seats"])}


def build():
    g = StateGraph(State)
    g.add_node("find_rules", find_rules)
    g.add_node("check_seats", check_seats)
    g.add_node("write_answer", write_answer)
    g.add_edge(START, "find_rules")
    g.add_edge("find_rules", "check_seats")
    g.add_conditional_edges("check_seats", after_check_seats, ["check_seats", "write_answer"])
    g.add_edge("write_answer", END)
    return g.compile()


def ask(question, flight):
    return build().invoke({"question": question, "flight": flight})
