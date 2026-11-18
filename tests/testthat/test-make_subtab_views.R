test_that("official views match the reference implementation when d is divisible by n_views", {
  v <- make_subtab_views(32, 4, 0.5, shuffle = FALSE)
  expect_equal(which(v[1, ]), 1:12)
  expect_equal(which(v[2, ]), 5:16)
  expect_equal(which(v[3, ]), 13:24)
  expect_equal(which(v[4, ]), 21:32)
  expect_true(all(rowSums(v) == 12))            # same width for every view
})

test_that("official views match the reference implementation when d is not divisible by n_views", {
  expect_warning(v <- make_subtab_views(34, 4, 0.5), "33, 34")

  expect_equal(which(v[1, ]), 1:12)
  expect_equal(which(v[2, ]), 5:16)
  expect_equal(which(v[3, ]), 13:24)
  expect_equal(which(v[4, ]), 21:32)
  expect_equal(unname(which(colSums(v) == 0)), c(33, 34))
})

test_that("the warning uses the column names when they are given", {
  names_d <- paste0("var", 1:10)
  expect_warning(make_subtab_views(10, 4, 0.5, column_names = names_d), "var9, var10")
})

test_that("overlap = 0 gives an exact partition when d is divisible by K", {
  v <- make_subtab_views(32, 4, 0)
  expect_true(all(colSums(v) == 1))
})

test_that("K = 1 gives one view with all columns", {
  expect_true(all(make_subtab_views(10, 1, 0.5)))
})

test_that("invalid arguments are rejected", {
  expect_error(make_subtab_views(5, 6, 0.5))                       # more views than columns
  expect_error(make_subtab_views(5, 0, 0.5))                       # no views
  expect_error(make_subtab_views(5, 2.5, 0.5))                     # n_views must be an integer
  expect_error(make_subtab_views(5, 2, 1.5))                       # overlap out of range
  expect_error(make_subtab_views(5, 2, -0.1))
  expect_error(make_subtab_views(0, 1, 0.5))                       # d must be positive
  expect_error(make_subtab_views(5.5, 2, 0.5))                     # d must be an integer
  expect_error(make_subtab_views(10, 2, 0.5, column_names = c("a", "b")), "column_names")
})
