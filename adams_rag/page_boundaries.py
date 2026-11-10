"""Recover physical pages without treating embedded control characters as pages."""


class PageBoundaryError(ValueError):
    pass


def retained_pdf_pages(body, metadata):
    count = metadata.get("ExtractedPageCount")
    if count is not None and (type(count) is not int or count < 1):
        raise PageBoundaryError("retained_page_count_invalid")
    lengths = metadata.get("RetainedPageLengths")
    if lengths is not None:
        if (not isinstance(lengths, list) or not lengths
                or any(type(n) is not int or n < 0 for n in lengths)
                or (count is not None and count != len(lengths))):
            raise PageBoundaryError("retained_page_map_invalid")
        pages, offset = [], 0
        for index, length in enumerate(lengths):
            end = offset + length
            if end > len(body):
                raise PageBoundaryError("retained_page_map_invalid")
            pages.append(body[offset:end])
            offset = end
            if index < len(lengths) - 1:
                if offset >= len(body) or body[offset] != "\f":
                    raise PageBoundaryError("retained_page_map_invalid")
                offset += 1
        if offset != len(body):
            raise PageBoundaryError("retained_page_map_invalid")
        return pages
    # Earlier one-page extractions already record an unambiguous physical count.
    if count == 1:
        return [body]
    pages = body.split("\f")
    if count is not None and count != len(pages):
        raise PageBoundaryError("retained_page_boundaries_ambiguous")
    # Legacy versions without a recorded count retain their previous convention.
    return pages
