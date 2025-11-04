// GopherData CLI - Command-line tool for data inspection and manipulation
package main

import (
	"fmt"
	"os"

	"github.com/TIVerse/GopherData/cmd/gopherdata/commands"
	"github.com/spf13/cobra"
)

var version = "1.0.0"

func main() {
	rootCmd := &cobra.Command{
		Use:   "gopherdata",
		Short: "GopherData CLI - Data inspection and manipulation tool",
		Long: `GopherData CLI provides command-line utilities for working with data files.
Supports CSV, JSON, and other formats for quick data exploration and conversion.`,
		Version: version,
	}

	// Add subcommands
	rootCmd.AddCommand(commands.InfoCmd())
	rootCmd.AddCommand(commands.HeadCmd())
	rootCmd.AddCommand(commands.TailCmd())
	rootCmd.AddCommand(commands.DescribeCmd())
	rootCmd.AddCommand(commands.ConvertCmd())
	rootCmd.AddCommand(commands.FilterCmd())
	rootCmd.AddCommand(commands.SelectCmd())

	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
