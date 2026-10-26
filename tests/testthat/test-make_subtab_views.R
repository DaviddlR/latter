test_that("official views match the reference implementation", {
  v <- make_subtab_views_official(32, 4, 0.5)
  expect_equal(which(v[1, ]), 1:12)
  expect_equal(which(v[2, ]), 5:16)
  expect_equal(which(v[3, ]), 13:24)
  expect_equal(which(v[4, ]), 21:32)
  expect_true(all(rowSums(v) == 12))            # same width for every view
})

test_that("overlap = 0 gives an exact partition when d is divisible by K", {
  v <- make_subtab_views_official(32, 4, 0)
  expect_true(all(colSums(v) == 1))
})

test_that("K = 1 gives one view with all columns", {
  expect_true(all(make_subtab_views_official(10, 1, 0.5)))
})

test_that("invalid arguments are rejected", {
  expect_error(make_subtab_views_official(5, 6, 0.5))
  expect_error(make_subtab_views_official(5, 2, 1.5))
})
