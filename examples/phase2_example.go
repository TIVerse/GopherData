//go:build ignore
// +build ignore

package main

import (
	"fmt"
	"log"

	"github.com/TIVerse/GopherData/core"
	"github.com/TIVerse/GopherData/dataframe"
	jsonIO "github.com/TIVerse/GopherData/io/json"
)

func main() {
	fmt.Println("=== GopherData Phase 2: Essential Operations Demo ===\n")

	// Create sample sales data
	salesData := map[string]any{
		"product_id": []int64{1, 1, 2, 2, 3, 3, 1, 2, 3, 1},
		"category":   []string{"Electronics", "Electronics", "Furniture", "Furniture", "Electronics", "Electronics", "Electronics", "Furniture", "Electronics", "Electronics"},
		"region":     []string{"North", "South", "North", "South", "North", "South", "North", "North", "South", "South"},
		"amount":     []float64{1200, 1500, 800, 900, 1100, 1300, 1400, 850, 1250, 1600},
		"quantity":   []int64{10, 12, 8, 9, 11, 13, 14, 8, 12, 15},
		"discount":   []any{0.1, nil, 0.15, 0.0, 0.05, nil, 0.0, 0.1, nil, 0.05},
	}

	sales, err := dataframe.New(salesData)
	if err != nil {
		log.Fatalf("Failed to create sales DataFrame: %v", err)
	}

	fmt.Println("Original Sales Data:")
	fmt.Println(sales.Head(5))
	fmt.Println()

	// =============================================================================
	// 1. Missing Data Handling
	// =============================================================================
	fmt.Println("=== 1. Missing Data Handling ===")
	
	// Fill missing discounts with 0
	sales = sales.FillNAColumn("discount", 0.0)
	fmt.Println("After filling missing discounts with 0:")
	fmt.Println(sales.Head(5))
	fmt.Println()

	// =============================================================================
	// 2. GroupBy Aggregation
	// =============================================================================
	fmt.Println("=== 2. GroupBy Aggregation ===")
	
	// Group by category and aggregate
	grouped, err := sales.GroupBy("category")
	if err != nil {
		log.Fatalf("GroupBy failed: %v", err)
	}

	categoryStats, err := grouped.Agg(map[string]string{
		"amount":   "sum",
		"quantity": "mean",
	})
	if err != nil {
		log.Fatalf("Aggregation failed: %v", err)
	}

	fmt.Println("Sales by Category:")
	fmt.Println(categoryStats)
	fmt.Println()

	// Multiple aggregations per column
	detailedStats, err := grouped.AggMultiple(map[string][]string{
		"amount": {"sum", "mean", "std"},
		"quantity": {"min", "max"},
	})
	if err != nil {
		log.Fatalf("Multiple aggregations failed: %v", err)
	}

	fmt.Println("Detailed Statistics by Category:")
	fmt.Println(detailedStats)
	fmt.Println()

	// =============================================================================
	// 3. Join Operations
	// =============================================================================
	fmt.Println("=== 3. Join Operations ===")
	
	// Create product information DataFrame
	productData := map[string]any{
		"product_id": []int64{1, 2, 3},
		"name":       []string{"Laptop", "Desk", "Monitor"},
		"cost":       []float64{800.0, 400.0, 600.0},
	}
	products, err := dataframe.New(productData)
	if err != nil {
		log.Fatalf("Failed to create products DataFrame: %v", err)
	}

	// Join sales with product information
	enrichedSales, err := sales.Join(products, dataframe.JoinInner, "product_id")
	if err != nil {
		log.Fatalf("Join failed: %v", err)
	}

	fmt.Println("Sales enriched with product information:")
	fmt.Println(enrichedSales.Head(5))
	fmt.Println()

	// =============================================================================
	// 4. Sort Operations
	// =============================================================================
	fmt.Println("=== 4. Sort Operations ===")
	
	// Sort by amount descending
	sortedSales := sales.Sort("amount", core.Descending)
	fmt.Println("Top sales by amount:")
	fmt.Println(sortedSales.Head(5))
	fmt.Println()

	// Multi-column sort
	multiSortSales := sales.SortMulti(
		[]string{"category", "amount"},
		[]core.Order{core.Ascending, core.Descending},
	)
	fmt.Println("Sales sorted by category (asc) then amount (desc):")
	fmt.Println(multiSortSales.Head(5))
	fmt.Println()

	// =============================================================================
	// 5. Window Functions
	// =============================================================================
	fmt.Println("=== 5. Window Functions ===")
	
	// Rolling mean with window size 3
	window := sales.Rolling(3)
	rollingMean, err := window.Mean("amount")
	if err != nil {
		log.Fatalf("Rolling mean failed: %v", err)
	}

	// Window functions return Series[float64], use separately
	// salesWithRolling := sales.WithColumn("rolling_avg", rollingMean)
	fmt.Println("Sales with 3-period rolling average:")
	fmt.Printf("Rolling mean calculated for %d rows\n", rollingMean.Len())
	fmt.Println()

	// Expanding sum
	expanding := sales.Expanding(1)
	expandingSum, err := expanding.Sum("amount")
	if err != nil {
		log.Fatalf("Expanding sum failed: %v", err)
	}

	// salesWithExpanding := sales.WithColumn("cumulative_sum", expandingSum)
	fmt.Println("Sales with cumulative sum:")
	fmt.Printf("Expanding sum calculated for %d rows\n", expandingSum.Len())
	fmt.Println()

	// Exponentially weighted moving average
	ewm := sales.EWM(0.3)
	ewmMean, err := ewm.Mean("amount")
	if err != nil {
		log.Fatalf("EWM failed: %v", err)
	}

	// salesWithEWM := sales.WithColumn("ewm_avg", ewmMean)
	fmt.Println("Sales with exponentially weighted moving average:")
	fmt.Printf("EWM calculated for %d rows\n", ewmMean.Len())
	fmt.Println()

	// =============================================================================
	// 6. Apply Operations
	// =============================================================================
	fmt.Println("=== 6. Apply Operations ===")
	
	// Calculate net amount after discount
	salesWithNet := sales.Apply(func(row *dataframe.Row) any {
		amount, _ := row.Get("amount")
		discount, _ := row.Get("discount")
		if amount == nil || discount == nil {
			return nil
		}
		return amount.(float64) * (1 - discount.(float64))
	}, "net_amount")

	fmt.Println("Sales with net amount after discount:")
	fmt.Println(salesWithNet.Head(5))
	fmt.Println()

	// Apply column transformation (double all amounts)
	doubledSales := sales.ApplyColumn("amount", func(val any) any {
		if val == nil {
			return nil
		}
		return val.(float64) * 2
	})

	fmt.Println("Sales with doubled amounts:")
	fmt.Println(doubledSales.Head(5))
	fmt.Println()

	// =============================================================================
	// 7. Reshape Operations
	// =============================================================================
	fmt.Println("=== 7. Reshape Operations ===")
	
	// Pivot table: product_id x region
	pivotTable, err := sales.Pivot("product_id", "region", "amount")
	if err != nil {
		log.Fatalf("Pivot failed: %v", err)
	}

	fmt.Println("Pivot table (product_id x region):")
	fmt.Println(pivotTable)
	fmt.Println()

	// Melt operation (wide to long format)
	meltedData, err := pivotTable.Melt([]string{"product_id"}, nil, "region", "amount")
	if err != nil {
		log.Fatalf("Melt failed: %v", err)
	}

	fmt.Println("Melted data (back to long format):")
	fmt.Println(meltedData.Head(5))
	fmt.Println()

	// =============================================================================
	// 8. Statistical Summary
	// =============================================================================
	fmt.Println("=== 8. Statistical Summary ===")
	
	stats, err := sales.Describe()
	if err != nil {
		log.Fatalf("Describe failed: %v", err)
	}

	fmt.Println("Statistical summary:")
	fmt.Println(stats)
	fmt.Println()

	// =============================================================================
	// 9. JSON I/O
	// =============================================================================
	fmt.Println("=== 9. JSON I/O ===")
	
	// Write to JSON (records format)
	recordsPath := "/tmp/sales_records.json"
	err = jsonIO.WriteJSON(sales, recordsPath)
	if err != nil {
		log.Fatalf("WriteJSON failed: %v", err)
	}
	fmt.Printf("Sales data written to %s (records format)\n", recordsPath)

	// Read back from JSON
	readSales, err := jsonIO.ReadJSON(recordsPath)
	if err != nil {
		log.Fatalf("ReadJSON failed: %v", err)
	}
	fmt.Printf("Sales data read from JSON: %d rows, %d cols\n", readSales.Nrows(), readSales.Ncols())
	fmt.Println()

	// Write to JSON (columns format)
	columnsPath := "/tmp/sales_columns.json"
	err = jsonIO.WriteJSON(sales, columnsPath, jsonIO.Orient("columns"))
	if err != nil {
		log.Fatalf("WriteJSON (columns) failed: %v", err)
	}
	fmt.Printf("Sales data written to %s (columns format)\n", columnsPath)

	// Write to JSONL (lines format)
	linesPath := "/tmp/sales.jsonl"
	err = jsonIO.WriteJSON(sales, linesPath, jsonIO.Lines())
	if err != nil {
		log.Fatalf("WriteJSON (lines) failed: %v", err)
	}
	fmt.Printf("Sales data written to %s (JSONL format)\n", linesPath)
	fmt.Println()

	// =============================================================================
	// 10. Complete Workflow Example
	// =============================================================================
	fmt.Println("=== 10. Complete Workflow Example ===")
	fmt.Println("Load → Clean → GroupBy → Join → Sort → Save")
	fmt.Println()

	// Step 1: Clean data
	cleaned := sales.DropNA().FillNA(0.0)
	fmt.Printf("Step 1 - Cleaned: %d rows\n", cleaned.Nrows())

	// Step 2: GroupBy aggregation
	grouped2, _ := cleaned.GroupBy("category")
	summary, _ := grouped2.Agg(map[string]string{
		"amount":   "sum",
		"quantity": "mean",
	})
	fmt.Printf("Step 2 - Grouped: %d categories\n", summary.Nrows())

	// Step 3: Join with product info
	result, _ := summary.Join(products, dataframe.JoinInner, "category")
	fmt.Printf("Step 3 - Joined: %d rows\n", result.Nrows())

	// Step 4: Sort by total sales
	finalResult := result.Sort("amount", core.Descending)
	fmt.Printf("Step 4 - Sorted by amount\n")

	// Step 5: Save result
	outputPath := "/tmp/sales_summary.json"
	err = jsonIO.WriteJSON(finalResult, outputPath)
	if err != nil {
		log.Fatalf("Failed to save result: %v", err)
	}
	fmt.Printf("Step 5 - Saved to %s\n", outputPath)
	fmt.Println()

	fmt.Println("Final Result:")
	fmt.Println(finalResult)
	
	fmt.Println("\n=== Phase 2 Demo Complete ===")
}
