# Ideas for improvements

- Generate an index on the fly
    - The index stores chunks with the min/max values of each attribute in that chunk
    - Chunks could be e.g. 10k points in size
    - While scanning first (without an index) we can create a chunk header for each chunk, but only for the queried attribute(s)
        - So if we query by bounds, we compute an AABB for each chunk. If we query by object class, we create a class histogram, and so on...
    - Upon further scans, we first consult the index to find the matching chunks
    - Within each chunk, we can potentially refine the chunk, e.g. by calculating first half and second half chunks, and compare them if they are sufficiently different
    - For uncompressed data, we could also restructure the data (i.e. sort the points), but this only works for a single attribute
- The search should work stream-based so that we can stream the results into whatever application we want
    - In particular relevant for rendering / sending the data to a remote application