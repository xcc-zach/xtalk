[**xtalk-client**](../globals.md)

***

[xtalk-client](../globals.md) / BaseDeferredTaskScheduler

# Abstract Class: BaseDeferredTaskScheduler

Defined in: bases/task-scheduler.ts:6

Schedules deferred tasks without exposing platform-specific timing primitives.

## Constructors

### Constructor

> **new BaseDeferredTaskScheduler**(): `BaseDeferredTaskScheduler`

#### Returns

`BaseDeferredTaskScheduler`

## Methods

### dispose()

> `abstract` **dispose**(): `void`

Defined in: bases/task-scheduler.ts:17

Stops the scheduler and clears any queued tasks that have not run yet.

#### Returns

`void`

***

### schedule()

> `abstract` **schedule**(`task`): `void`

Defined in: bases/task-scheduler.ts:12

Enqueues a task to run asynchronously after the current hot path completes.

#### Parameters

##### task

() => `void`

Task to run in a later task turn.

#### Returns

`void`
