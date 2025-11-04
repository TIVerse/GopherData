// Package parallel provides concurrency utilities for parallel data processing.
package parallel

import (
	"context"
	"runtime"
	"sync"
)

// Pool represents a worker pool for parallel task execution.
type Pool struct {
	workers int
	tasks   chan func()
	wg      sync.WaitGroup
	ctx     context.Context
	cancel  context.CancelFunc
}

// NewPool creates a new worker pool with the specified number of workers.
// If workers <= 0, it defaults to runtime.NumCPU().
func NewPool(workers int) *Pool {
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	
	p := &Pool{
		workers: workers,
		tasks:   make(chan func(), workers*2), // Buffer size: workers * 2
		ctx:     ctx,
		cancel:  cancel,
	}
	
	// Start worker goroutines
	for i := 0; i < workers; i++ {
		go p.worker()
	}
	
	return p
}

// worker processes tasks from the task channel.
func (p *Pool) worker() {
	for {
		select {
		case task := <-p.tasks:
			task()
			p.wg.Done()
		case <-p.ctx.Done():
			return
		}
	}
}

// Submit submits a task to the pool for execution.
func (p *Pool) Submit(fn func()) {
	p.wg.Add(1)
	p.tasks <- fn
}

// Wait waits for all submitted tasks to complete.
func (p *Pool) Wait() {
	p.wg.Wait()
}

// Close closes the pool and stops all workers.
func (p *Pool) Close() {
	p.cancel()
	close(p.tasks)
}

// ParallelMap applies a function to each element in parallel and returns results.
func ParallelMap[T, R any](data []T, fn func(T) R, workers int) []R {
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	
	if len(data) == 0 {
		return []R{}
	}
	
	result := make([]R, len(data))
	chunkSize := (len(data) + workers - 1) / workers
	
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if start >= len(data) {
			break
		}
		if end > len(data) {
			end = len(data)
		}
		
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				result[i] = fn(data[i])
			}
		}(start, end)
	}
	
	wg.Wait()
	return result
}

// ParallelReduce reduces data using a function in parallel.
func ParallelReduce[T, R any](data []T, fn func(T, R) R, initial R, combiner func(R, R) R, workers int) R {
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	
	if len(data) == 0 {
		return initial
	}
	
	chunkSize := (len(data) + workers - 1) / workers
	partialResults := make([]R, workers)
	
	var wg sync.WaitGroup
	actualWorkers := 0
	
	for w := 0; w < workers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if start >= len(data) {
			break
		}
		if end > len(data) {
			end = len(data)
		}
		
		actualWorkers++
		wg.Add(1)
		go func(s, e, idx int) {
			defer wg.Done()
			result := initial
			for i := s; i < e; i++ {
				result = fn(data[i], result)
			}
			partialResults[idx] = result
		}(start, end, w)
	}
	
	wg.Wait()
	
	// Combine partial results
	final := initial
	for i := 0; i < actualWorkers; i++ {
		final = combiner(final, partialResults[i])
	}
	
	return final
}

// ParallelForEach applies a function to each element in parallel (no return value).
func ParallelForEach[T any](data []T, fn func(T), workers int) {
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	
	if len(data) == 0 {
		return
	}
	
	chunkSize := (len(data) + workers - 1) / workers
	
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if start >= len(data) {
			break
		}
		if end > len(data) {
			end = len(data)
		}
		
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				fn(data[i])
			}
		}(start, end)
	}
	
	wg.Wait()
}
