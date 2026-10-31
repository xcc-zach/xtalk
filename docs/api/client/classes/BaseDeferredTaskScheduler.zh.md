[**xtalk-client**](../globals.zh.md)

***

[xtalk-client](../globals.zh.md) / BaseDeferredTaskScheduler

# 抽象类： BaseDeferredTaskScheduler

定义于： bases/task-scheduler.ts:6

调度延迟任务，而不暴露平台相关的计时原语。

## 构造函数

### 构造函数

> **new BaseDeferredTaskScheduler**(): `BaseDeferredTaskScheduler`

#### 返回值

`BaseDeferredTaskScheduler`

## 方法

### dispose()

> `abstract` **dispose**(): `void`

定义于： bases/task-scheduler.ts:17

停止调度器并清除尚未执行的排队任务。

#### 返回值

`void`

***

### schedule()

> `abstract` **schedule**(`task`): `void`

定义于： bases/task-scheduler.ts:12

将任务排入队列，在当前热路径完成后异步执行。

#### 参数

##### task

() => `void`

在后续任务轮次中执行的任务。

#### 返回值

`void`
