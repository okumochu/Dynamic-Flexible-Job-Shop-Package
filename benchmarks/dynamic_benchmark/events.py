from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Union, Optional


class EventType(Enum):
    NEW_JOB = auto()
    MACHINE_BREAKDOWN = auto()
    PRIORITY_CHANGE = auto()


@dataclass
class NewJobArrival:
    """New job arrival event.

    Attributes:
        time: Event time
        job_id: The id that the job will take in the system
    """
    time: float
    job_id: int
    # The arriving job object (populated by the generator)
    job: Optional["Job"] = None


@dataclass
class MachineBreakdown:
    """Machine breakdown event.

    Attributes:
        time: Breakdown start time
        machine_id: Affected machine id
        repair_duration: Duration until the machine is restored
    """
    time: float
    machine_id: int
    repair_duration: float


@dataclass
class PriorityChange:
    """Priority change event (weight change for a job).

    Attributes:
        time: Event time
        job_id: Job id whose weight changes
        new_weight: The new weight to assign
    """
    time: float
    job_id: int
    new_weight: int


Event = Union[NewJobArrival, MachineBreakdown, PriorityChange]


