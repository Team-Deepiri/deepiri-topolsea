# Filter JSON dialect

Topolsea metadata filters are JSON objects parsed by `Filter::from_json`.

## Equality

```json
{ "tag": "alpha" }
{ "tag": { "$eq": "alpha" } }
```

## Comparison

Numeric or string-ordered comparisons:

```json
{ "score": { "$gt": 0.5 } }
{ "score": { "$gte": 0.5 } }
{ "score": { "$lt": 10 } }
{ "score": { "$lte": 10 } }
{ "tag": { "$ne": "alpha" } }
```

## Membership

```json
{ "tag": { "$in": ["alpha", "beta"] } }
```

## Combinators

```json
{
  "$and": [
    { "tag": { "$ne": "spam" } },
    { "score": { "$gte": 0.2 } }
  ]
}
```

```json
{
  "$or": [
    { "lang": "en" },
    { "lang": "es" }
  ]
}
```

Multi-key objects are an implicit `$and`:

```json
{ "tag": "alpha", "score": { "$gt": 1 } }
```

## Payload-aware ANN (Phase A4)

Equality, `$ne`, `$in`, and boolean `$and`/`$or` of those ops are resolved via an inverted index (roaring bitmaps) and constrain HNSW / Flat / Z-Column candidate generation.

Range ops (`$gt`/`$gte`/`$lt`/`$lte`) currently fall back to post-filtering with overfetch (`top_k × 10`).
