package core

// Option represents a functional option pattern for configuring types.
// T is the type being configured.
type Option[T any] func(*T) error

// ApplyOptions applies a slice of options to a target.
// Returns the first error encountered, if any.
func ApplyOptions[T any](target *T, opts ...Option[T]) error {
	for _, opt := range opts {
		if err := opt(target); err != nil {
			return err
		}
	}
	return nil
}
