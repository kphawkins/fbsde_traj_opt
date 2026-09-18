
Notes for code style
* Use trailing return syntax for all functions
* Use `noexcept` by default
* Put `const` on the left
* 120 character lines
* Represent all indices with `std::size_t`
* To enforce invariants of classes use a `Make` factory constructor pattern which returns a `Result`
  or an error if the invariants would not be satisfied by the inputs.
