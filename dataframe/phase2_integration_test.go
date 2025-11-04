package dataframe

import (
	"testing"

	"github.com/TIVerse/GopherData/core"
)

// TestPhase2Integration tests a complete Phase 2 workflow
func TestPhase2Integration(t *testing.T) {
	// Create sample sales data
	salesData := map[string]any{
		"product_id": []int64{1, 1, 2, 2, 3, 3, 1, 2},
		"category":   []string{"A", "A", "B", "B", "A", "A", "A", "B"},
		"amount":     []float64{100.0, 150.0, 200.0, 180.0, 120.0, 130.0, 110.0, 190.0},
		"quantity":   []int64{10, 15, 20, 18, 12, 13, 11, 19},
		"discount":   []float64{0.1, 0.0, 0.15, 0.0, 0.05, 0.0, 0.0, 0.1},
	}

	df, err := New(salesData)
	if err != nil {
		t.Fatalf("Failed to create DataFrame: %v", err)
	}

	// Test 1: Clean data - handle missing values
	t.Run("MissingDataHandling", func(t *testing.T) {
		// FillNA with 0
		cleanedDf := df.FillNAColumn("discount", 0.0)
		if cleanedDf.Nrows() != df.Nrows() {
			t.Errorf("FillNA changed number of rows")
		}
	})

	// Test 2: GroupBy aggregation
	t.Run("GroupByAggregation", func(t *testing.T) {
		grouped, err := df.GroupBy("category")
		if err != nil {
			t.Fatalf("GroupBy failed: %v", err)
		}

		result, err := grouped.Agg(map[string]string{
			"amount":   "sum",
			"quantity": "mean",
		})
		if err != nil {
			t.Fatalf("Aggregation failed: %v", err)
		}

		// Should have 2 groups (A and B)
		if result.Nrows() != 2 {
			t.Errorf("Expected 2 groups, got %d", result.Nrows())
		}
	})

	// Test 3: Multiple aggregations per column
	t.Run("MultipleAggregations", func(t *testing.T) {
		grouped, err := df.GroupBy("product_id")
		if err != nil {
			t.Fatalf("GroupBy failed: %v", err)
		}

		result, err := grouped.AggMultiple(map[string][]string{
			"amount":   {"sum", "mean", "std"},
			"quantity": {"min", "max"},
		})
		if err != nil {
			t.Fatalf("AggMultiple failed: %v", err)
		}

		// Should have 3 product_ids
		if result.Nrows() != 3 {
			t.Errorf("Expected 3 product groups, got %d", result.Nrows())
		}

		// Check result columns
		expectedCols := []string{"product_id", "amount_sum", "amount_mean", "amount_std", "quantity_min", "quantity_max"}
		resultCols := result.Columns()
		if len(resultCols) != len(expectedCols) {
			t.Errorf("Expected %d columns, got %d", len(expectedCols), len(resultCols))
		}
	})

	// Test 4: Join operations
	t.Run("JoinOperations", func(t *testing.T) {
		// Create product info DataFrame
		productData := map[string]any{
			"product_id": []int64{1, 2, 3, 4},
			"name":       []string{"Widget", "Gadget", "Doohickey", "Thingamajig"},
			"price":      []float64{50.0, 75.0, 60.0, 80.0},
		}
		products, err := New(productData)
		if err != nil {
			t.Fatalf("Failed to create products DataFrame: %v", err)
		}

		// Inner join
		innerResult, err := df.Join(products, JoinInner, "product_id")
		if err != nil {
			t.Fatalf("Inner join failed: %v", err)
		}
		if innerResult.Nrows() != 8 {
			t.Errorf("Inner join: expected 8 rows, got %d", innerResult.Nrows())
		}

		// Left join
		leftResult, err := df.Join(products, JoinLeft, "product_id")
		if err != nil {
			t.Fatalf("Left join failed: %v", err)
		}
		if leftResult.Nrows() != 8 {
			t.Errorf("Left join: expected 8 rows, got %d", leftResult.Nrows())
		}
	})

	// Test 5: Sort operations
	t.Run("SortOperations", func(t *testing.T) {
		// Sort by amount descending
		sortedDf := df.Sort("amount", core.Descending)
		if sortedDf.Nrows() != df.Nrows() {
			t.Errorf("Sort changed number of rows")
		}

		// Multi-column sort
		multiSortDf := df.SortMulti(
			[]string{"category", "amount"},
			[]core.Order{core.Ascending, core.Descending},
		)
		if multiSortDf.Nrows() != df.Nrows() {
			t.Errorf("Multi-column sort changed number of rows")
		}
	})

	// Test 6: Window functions
	t.Run("WindowFunctions", func(t *testing.T) {
		// Rolling mean
		window := df.Rolling(3)
		rollingMean, err := window.Mean("amount")
		if err != nil {
			t.Fatalf("Rolling mean failed: %v", err)
		}
		if rollingMean.Len() != df.Nrows() {
			t.Errorf("Rolling mean: expected %d values, got %d", df.Nrows(), rollingMean.Len())
		}

		// Expanding sum
		expanding := df.Expanding(1)
		expandingSum, err := expanding.Sum("amount")
		if err != nil {
			t.Fatalf("Expanding sum failed: %v", err)
		}
		if expandingSum.Len() != df.Nrows() {
			t.Errorf("Expanding sum: expected %d values, got %d", df.Nrows(), expandingSum.Len())
		}
	})

	// Test 7: Apply operations
	t.Run("ApplyOperations", func(t *testing.T) {
		// Apply function to calculate total with discount
		resultDf := df.Apply(func(row *Row) any {
			amount, _ := row.Get("amount")
			discount, _ := row.Get("discount")
			if amount == nil || discount == nil {
				return nil
			}
			return amount.(float64) * (1 - discount.(float64))
		}, "total")

		if resultDf.Ncols() != df.Ncols()+1 {
			t.Errorf("Apply should add one column")
		}

		// ApplyColumn - double all amounts
		doubledDf := df.ApplyColumn("amount", func(val any) any {
			if val == nil {
				return nil
			}
			return val.(float64) * 2
		})
		if doubledDf.Nrows() != df.Nrows() {
			t.Errorf("ApplyColumn changed number of rows")
		}
	})

	// Test 8: Reshape operations
	t.Run("ReshapeOperations", func(t *testing.T) {
		// Pivot table
		pivotDf, err := df.Pivot("product_id", "category", "amount")
		if err != nil {
			t.Fatalf("Pivot failed: %v", err)
		}
		if pivotDf.Nrows() != 3 {
			t.Errorf("Pivot: expected 3 rows (product_ids), got %d", pivotDf.Nrows())
		}

		// Melt operation
		meltedDf, err := pivotDf.Melt([]string{"product_id"}, nil, "category", "amount")
		if err != nil {
			t.Fatalf("Melt failed: %v", err)
		}
		if meltedDf.Nrows() == 0 {
			t.Errorf("Melt produced no rows")
		}
	})

	// Test 9: Aggregations
	t.Run("Aggregations", func(t *testing.T) {
		sums, err := df.Sum("amount")
		if err != nil {
			t.Fatalf("Sum failed: %v", err)
		}
		if len(sums) != 1 {
			t.Errorf("Sum: expected 1 result, got %d", len(sums))
		}

		means, err := df.Mean("amount", "quantity")
		if err != nil {
			t.Fatalf("Mean failed: %v", err)
		}
		if len(means) != 2 {
			t.Errorf("Mean: expected 2 results, got %d", len(means))
		}

		stats, err := df.Describe()
		if err != nil {
			t.Fatalf("Describe failed: %v", err)
		}
		if stats.Nrows() == 0 {
			t.Errorf("Describe: got 0 rows")
		}
	})
}

// TestGroupByEdgeCases tests GroupBy with edge cases
func TestGroupByEdgeCases(t *testing.T) {
	t.Run("EmptyGroups", func(t *testing.T) {
		data := map[string]any{
			"key": []int64{1, 1, 1},
			"val": []float64{1.0, 2.0, 3.0},
		}
		df, _ := New(data)

		grouped, err := df.GroupBy("key")
		if err != nil {
			t.Fatalf("GroupBy failed: %v", err)
		}

		result, err := grouped.Agg(map[string]string{"val": "sum"})
		if err != nil {
			t.Fatalf("Aggregation failed: %v", err)
		}

		if result.Nrows() != 1 {
			t.Errorf("Expected 1 group, got %d", result.Nrows())
		}
	})

	t.Run("NullKeys", func(t *testing.T) {
		data := map[string]any{
			"key": []any{1, nil, 2, nil, 1},
			"val": []float64{1.0, 2.0, 3.0, 4.0, 5.0},
		}
		df, _ := New(data)

		grouped, err := df.GroupBy("key")
		if err != nil {
			t.Fatalf("GroupBy failed: %v", err)
		}

		result, err := grouped.Agg(map[string]string{"val": "sum"})
		if err != nil {
			t.Fatalf("Aggregation failed: %v", err)
		}

		// Should have 3 groups: 1, 2, and null
		if result.Nrows() != 3 {
			t.Errorf("Expected 3 groups (including null), got %d", result.Nrows())
		}
	})

	t.Run("MultiColumnGroupBy", func(t *testing.T) {
		data := map[string]any{
			"key1": []string{"A", "A", "B", "B"},
			"key2": []int64{1, 2, 1, 2},
			"val":  []float64{10.0, 20.0, 30.0, 40.0},
		}
		df, _ := New(data)

		grouped, err := df.GroupBy("key1", "key2")
		if err != nil {
			t.Fatalf("GroupBy failed: %v", err)
		}

		result, err := grouped.Agg(map[string]string{"val": "sum"})
		if err != nil {
			t.Fatalf("Aggregation failed: %v", err)
		}

		// Should have 4 groups
		if result.Nrows() != 4 {
			t.Errorf("Expected 4 groups, got %d", result.Nrows())
		}
	})
}

// TestJoinEdgeCases tests join operations with edge cases
func TestJoinEdgeCases(t *testing.T) {
	t.Run("JoinWithNullKeys", func(t *testing.T) {
		left := map[string]any{
			"key": []any{1, nil, 3},
			"val": []string{"a", "b", "c"},
		}
		right := map[string]any{
			"key":  []any{1, nil, 2},
			"data": []string{"x", "y", "z"},
		}

		leftDf, _ := New(left)
		rightDf, _ := New(right)

		// Inner join should exclude nulls
		result, err := leftDf.Join(rightDf, JoinInner, "key")
		if err != nil {
			t.Fatalf("Join failed: %v", err)
		}

		// Only key=1 matches (nulls don't match)
		if result.Nrows() != 1 {
			t.Errorf("Inner join with nulls: expected 1 row, got %d", result.Nrows())
		}
	})

	t.Run("ManyToManyJoin", func(t *testing.T) {
		left := map[string]any{
			"key": []int64{1, 1, 2},
			"val": []string{"a", "b", "c"},
		}
		right := map[string]any{
			"key":  []int64{1, 1, 2},
			"data": []string{"x", "y", "z"},
		}

		leftDf, _ := New(left)
		rightDf, _ := New(right)

		result, err := leftDf.Join(rightDf, JoinInner, "key")
		if err != nil {
			t.Fatalf("Join failed: %v", err)
		}

		// 2x2 for key=1, 1x1 for key=2 = 5 rows
		if result.Nrows() != 5 {
			t.Errorf("Many-to-many join: expected 5 rows, got %d", result.Nrows())
		}
	})

	t.Run("OuterJoin", func(t *testing.T) {
		left := map[string]any{
			"key": []int64{1, 2, 3},
			"val": []string{"a", "b", "c"},
		}
		right := map[string]any{
			"key":  []int64{2, 3, 4},
			"data": []string{"x", "y", "z"},
		}

		leftDf, _ := New(left)
		rightDf, _ := New(right)

		result, err := leftDf.Join(rightDf, JoinOuter, "key")
		if err != nil {
			t.Fatalf("Outer join failed: %v", err)
		}

		// All unique keys: 1, 2, 3, 4 = 4 rows
		if result.Nrows() != 4 {
			t.Errorf("Outer join: expected 4 rows, got %d", result.Nrows())
		}
	})

	t.Run("CrossJoin", func(t *testing.T) {
		left := map[string]any{
			"id":  []int64{1, 2},
			"val": []string{"a", "b"},
		}
		right := map[string]any{
			"num":  []int64{10, 20, 30},
			"data": []string{"x", "y", "z"},
		}

		leftDf, _ := New(left)
		rightDf, _ := New(right)

		result, err := leftDf.Merge(rightDf, JoinCross, nil, nil)
		if err != nil {
			t.Fatalf("Cross join failed: %v", err)
		}

		// 2 * 3 = 6 rows
		if result.Nrows() != 6 {
			t.Errorf("Cross join: expected 6 rows, got %d", result.Nrows())
		}
	})
}

// TestWindowEdgeCases tests window functions with edge cases
func TestWindowEdgeCases(t *testing.T) {
	t.Run("RollingWithNulls", func(t *testing.T) {
		data := map[string]any{
			"val": []any{1.0, nil, 3.0, 4.0, nil, 6.0},
		}
		df, _ := New(data)

		window := df.Rolling(3, MinPeriods(1))
		result, err := window.Mean("val")
		if err != nil {
			t.Fatalf("Rolling mean failed: %v", err)
		}

		if result.Len() != 6 {
			t.Errorf("Rolling mean: expected 6 values, got %d", result.Len())
		}
	})

	t.Run("CenteredWindow", func(t *testing.T) {
		data := map[string]any{
			"val": []float64{1.0, 2.0, 3.0, 4.0, 5.0},
		}
		df, _ := New(data)

		window := df.Rolling(3, Center())
		result, err := window.Mean("val")
		if err != nil {
			t.Fatalf("Centered rolling mean failed: %v", err)
		}

		if result.Len() != 5 {
			t.Errorf("Centered rolling mean: expected 5 values, got %d", result.Len())
		}
	})

	t.Run("ExpandingWindow", func(t *testing.T) {
		data := map[string]any{
			"val": []float64{1.0, 2.0, 3.0, 4.0, 5.0},
		}
		df, _ := New(data)

		window := df.Expanding(2)
		result, err := window.Sum("val")
		if err != nil {
			t.Fatalf("Expanding sum failed: %v", err)
		}

		if result.Len() != 5 {
			t.Errorf("Expanding sum: expected 5 values, got %d", result.Len())
		}
	})
}
