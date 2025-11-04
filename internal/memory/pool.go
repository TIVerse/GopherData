// Package memory provides memory management utilities including pooling.
package memory

import "sync"

// Pool is a generic memory pool for reusing objects.
type Pool[T any] struct {
	pool sync.Pool
	new  func() T
}

// NewPool creates a new memory pool with the given constructor function.
func NewPool[T any](newFn func() T) *Pool[T] {
	return &Pool[T]{
		pool: sync.Pool{
			New: func() any {
				return newFn()
			},
		},
		new: newFn,
	}
}

// Get retrieves an object from the pool.
func (p *Pool[T]) Get() T {
	return p.pool.Get().(T)
}

// Put returns an object to the pool for reuse.
func (p *Pool[T]) Put(item T) {
	p.pool.Put(item)
}

// Pre-defined pools for common types

// ByteSlicePool is a pool for byte slices (4KB default capacity).
var ByteSlicePool = NewPool(func() []byte {
	return make([]byte, 0, 4096)
})

// IntSlicePool is a pool for int slices (1024 default capacity).
var IntSlicePool = NewPool(func() []int {
	return make([]int, 0, 1024)
})

// Float64SlicePool is a pool for float64 slices (1024 default capacity).
var Float64SlicePool = NewPool(func() []float64 {
	return make([]float64, 0, 1024)
})

// StringSlicePool is a pool for string slices (512 default capacity).
var StringSlicePool = NewPool(func() []string {
	return make([]string, 0, 512)
})

// MapStringAnyPool is a pool for map[string]any (256 default capacity).
var MapStringAnyPool = NewPool(func() map[string]any {
	return make(map[string]any, 256)
})

// Buffer is a reusable buffer for efficient I/O operations.
type Buffer struct {
	data []byte
	pos  int
}

// NewBuffer creates a new buffer with the specified size.
func NewBuffer(size int) *Buffer {
	return &Buffer{
		data: make([]byte, size),
		pos:  0,
	}
}

// Write writes data to the buffer.
func (b *Buffer) Write(p []byte) (int, error) {
	// Expand buffer if needed
	needed := b.pos + len(p)
	if needed > len(b.data) {
		newSize := len(b.data) * 2
		for newSize < needed {
			newSize *= 2
		}
		newData := make([]byte, newSize)
		copy(newData, b.data[:b.pos])
		b.data = newData
	}
	
	n := copy(b.data[b.pos:], p)
	b.pos += n
	return n, nil
}

// Reset resets the buffer for reuse.
func (b *Buffer) Reset() {
	b.pos = 0
}

// Bytes returns the current buffer contents.
func (b *Buffer) Bytes() []byte {
	return b.data[:b.pos]
}

// Len returns the current length of data in the buffer.
func (b *Buffer) Len() int {
	return b.pos
}

// Cap returns the capacity of the buffer.
func (b *Buffer) Cap() int {
	return len(b.data)
}

// BufferPool is a pool for reusable buffers.
var BufferPool = NewPool(func() *Buffer {
	return NewBuffer(8192) // 8KB default
})

// GetBuffer gets a buffer from the pool.
func GetBuffer() *Buffer {
	buf := BufferPool.Get()
	buf.Reset()
	return buf
}

// PutBuffer returns a buffer to the pool.
func PutBuffer(buf *Buffer) {
	if buf.Cap() <= 1024*1024 { // Only pool buffers <= 1MB
		BufferPool.Put(buf)
	}
}
