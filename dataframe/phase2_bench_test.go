package dataframe

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/TIVerse/GopherData/core"
)

// generateTestData creates a DataFrame with specified rows for benchmarking
func generateTestData(nrows int, seed int64) *DataFrame {
	r := rand.New(rand.NewSource(seed))
	
	data := map[string]any{
		"id":       make([]int64, nrows),
		"category": make([]string, nrows),
		"value1":   make([]float64, nrows),
		"value2":   make([]float64, nrows),
		"group":    make([]int64, nrows),
	}
	
	categories := []string{"A", "B", "C", "D", "E"}
	
	for i := 0; i < nrows; i++ {
		data["id"].([]int64)[i] = int64(i)
		data["category"].([]string)[i] = categories[i%len(categories)]
		data["value1"].([]float64)[i] = r.Float64() * 1000
		data["value2"].([]float64)[i] = r.Float64() * 100
		data["group"].([]int64)[i] = int64(r.Intn(100))
	}
	
	df, _ := New(data)
	return df
}

// BenchmarkGroupByAgg benchmarks GroupBy aggregation
func BenchmarkGroupByAgg(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("rows_%d", size), func(b *testing.B) {
			df := generateTestData(size, 42)
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				grouped, _ := df.GroupBy("category")
				_, _ = grouped.Agg(map[string]string{
					"value1": "sum",
					"value2": "mean",
				})
			}
		})
	}
}

// BenchmarkGroupByMultipleAgg benchmarks multiple aggregations per column
func BenchmarkGroupByMultipleAgg(b *testing.B) {
	df := generateTestData(100000, 42)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		grouped, _ := df.GroupBy("category")
		_, _ = grouped.AggMultiple(map[string][]string{
			"value1": {"sum", "mean", "std"},
			"value2": {"min", "max", "median"},
		})
	}
}

// BenchmarkGroupByManyGroups benchmarks GroupBy with many groups
func BenchmarkGroupByManyGroups(b *testing.B) {
	df := generateTestData(1000000, 42)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		grouped, _ := df.GroupBy("group")
		_, _ = grouped.Agg(map[string]string{
			"value1": "sum",
		})
	}
}

// BenchmarkJoinInner benchmarks inner join operations
func BenchmarkJoinInner(b *testing.B) {
	sizes := []struct {
		left  int
		right int
	}{
		{1000, 1000},
		{10000, 10000},
		{100000, 100000},
		{1000000, 100000},
	}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("left_%d_right_%d", size.left, size.right), func(b *testing.B) {
			left := generateTestData(size.left, 42)
			right := generateTestData(size.right, 43)
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = left.Join(right, JoinInner, "category")
			}
		})
	}
}

// BenchmarkJoinLeft benchmarks left join operations
func BenchmarkJoinLeft(b *testing.B) {
	left := generateTestData(100000, 42)
	right := generateTestData(10000, 43)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = left.Join(right, JoinLeft, "category")
	}
}

// BenchmarkJoinOuter benchmarks outer join operations
func BenchmarkJoinOuter(b *testing.B) {
	left := generateTestData(100000, 42)
	right := generateTestData(100000, 43)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = left.Join(right, JoinOuter, "category")
	}
}

// BenchmarkSort benchmarks single-column sorting
func BenchmarkSort(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("rows_%d", size), func(b *testing.B) {
			df := generateTestData(size, 42)
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = df.Sort("value1", core.Ascending)
			}
		})
	}
}

// BenchmarkSortMulti benchmarks multi-column sorting
func BenchmarkSortMulti(b *testing.B) {
	df := generateTestData(100000, 42)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = df.SortMulti(
			[]string{"category", "value1"},
			[]core.Order{core.Ascending, core.Descending},
		)
	}
}

// BenchmarkRollingMean benchmarks rolling mean calculation
func BenchmarkRollingMean(b *testing.B) {
	sizes := []int{1000, 10000, 100000, 1000000}
	windowSizes := []int{10, 50, 100}
	
	for _, size := range sizes {
		for _, winSize := range windowSizes {
			b.Run(fmt.Sprintf("rows_%d_window_%d", size, winSize), func(b *testing.B) {
				df := generateTestData(size, 42)
				
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					window := df.Rolling(winSize)
					_, _ = window.Mean("value1")
				}
			})
		}
	}
}

// BenchmarkRollingStd benchmarks rolling standard deviation
func BenchmarkRollingStd(b *testing.B) {
	df := generateTestData(100000, 42)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		window := df.Rolling(50)
		_, _ = window.Std("value1")
	}
}

// BenchmarkExpandingSum benchmarks expanding window sum
func BenchmarkExpandingSum(b *testing.B) {
	df := generateTestData(100000, 42)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		window := df.Expanding(1)
		_, _ = window.Sum("value1")
	}
}

// BenchmarkEWM benchmarks exponentially weighted moving average
func BenchmarkEWM(b *testing.B) {
	df := generateTestData(100000, 42)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		window := df.EWM(0.2)
		_, _ = window.Mean("value1")
	}
}

// BenchmarkApply benchmarks row-wise apply operation
func BenchmarkApply(b *testing.B) {
	sizes := []int{1000, 10000, 100000}
	
	for _, size := range sizes {
		b.Run(fmt.Sprintf("rows_%d", size), func(b *testing.B) {
			df := generateTestData(size, 42)
			
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = df.Apply(func(row *Row) any {
					val1, _ := row.Get("value1")
					val2, _ := row.Get("value2")
					if val1 == nil || val2 == nil {
						return nil
					}
					return val1.(float64) + val2.(float64)
				}, "sum")
			}
		})
	}
}

// BenchmarkApplyColumn benchmarks column-wise apply
func BenchmarkApplyColumn(b *testing.B) {
	df := generateTestData(100000, 42)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = df.ApplyColumn("value1", func(val any) any {
			if val == nil {
				return nil
			}
			return val.(float64) * 2
		})
	}
}

// BenchmarkInterpolate benchmarks interpolation methods
func BenchmarkInterpolate(b *testing.B) {
	// Create data with nulls
	data := map[string]any{
		"val": make([]any, 10000),
	}
	for i := 0; i < 10000; i++ {
		if i%5 == 0 {
			data["val"].([]any)[i] = nil
		} else {
			data["val"].([]any)[i] = float64(i)
		}
	}
	df, _ := New(data)
	
	methods := []string{"linear", "ffill", "bfill"}
	
	for _, method := range methods {
		b.Run(method, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = df.Interpolate(method)
			}
		})
	}
}

// BenchmarkDropNA benchmarks null value removal
func BenchmarkDropNA(b *testing.B) {
	// Create data with nulls
	data := map[string]any{
		"val1": make([]any, 100000),
		"val2": make([]any, 100000),
	}
	for i := 0; i < 100000; i++ {
		if i%10 == 0 {
			data["val1"].([]any)[i] = nil
		} else {
			data["val1"].([]any)[i] = float64(i)
		}
		if i%15 == 0 {
			data["val2"].([]any)[i] = nil
		} else {
			data["val2"].([]any)[i] = float64(i * 2)
		}
	}
	df, _ := New(data)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = df.DropNA()
	}
}

// BenchmarkFillNA benchmarks null value filling
func BenchmarkFillNA(b *testing.B) {
	// Create data with nulls
	data := map[string]any{
		"val": make([]any, 100000),
	}
	for i := 0; i < 100000; i++ {
		if i%10 == 0 {
			data["val"].([]any)[i] = nil
		} else {
			data["val"].([]any)[i] = float64(i)
		}
	}
	df, _ := New(data)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = df.FillNA(0.0)
	}
}

// BenchmarkPivot benchmarks pivot table creation
func BenchmarkPivot(b *testing.B) {
	df := generateTestData(10000, 42)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = df.Pivot("id", "category", "value1")
	}
}

// BenchmarkMelt benchmarks wide-to-long transformation
func BenchmarkMelt(b *testing.B) {
	df := generateTestData(10000, 42)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = df.Melt([]string{"id"}, []string{"value1", "value2"}, "variable", "value")
	}
}

// BenchmarkDescribe benchmarks statistical summary generation
func BenchmarkDescribe(b *testing.B) {
	df := generateTestData(100000, 42)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = df.Describe()
	}
}

// BenchmarkAggregations benchmarks individual aggregation functions
func BenchmarkAggregations(b *testing.B) {
	df := generateTestData(1000000, 42)
	
	b.Run("Sum", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = df.Sum("value1")
		}
	})
	
	b.Run("Mean", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = df.Mean("value1")
		}
	})
	
	b.Run("Median", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = df.Median("value1")
		}
	})
	
	b.Run("Std", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = df.Std("value1")
		}
	})
	
	b.Run("Min", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = df.Min("value1")
		}
	})
	
	b.Run("Max", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = df.Max("value1")
		}
	})
}

// BenchmarkMemoryEfficiency measures memory usage patterns
func BenchmarkMemoryEfficiency(b *testing.B) {
	b.Run("SelectColumns", func(b *testing.B) {
		df := generateTestData(100000, 42)
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = df.Select("id", "value1")
		}
	})
	
	b.Run("FilterRows", func(b *testing.B) {
		df := generateTestData(100000, 42)
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = df.Filter(func(row *Row) bool {
				val, _ := row.Get("value1")
				if val == nil {
					return false
				}
				return val.(float64) > 500
			})
		}
	})
	
	b.Run("Copy", func(b *testing.B) {
		df := generateTestData(100000, 42)
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = df.Copy()
		}
	})
}
