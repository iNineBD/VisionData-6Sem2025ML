SELECT
    t.TicketKey,
    t.QtTickets,
    -- entrada
    entry.Year AS EntryYear,
    entry.Month AS EntryMonth,
    entry.Day AS EntryDay,
    -- primeira resposta
    response.Year AS ResponseYear,
    response.Month AS ResponseMonth,
    response.Day AS ResponseDay,
    -- fechamento
    closed.Year AS ClosedYear,
    closed.Month AS ClosedMonth,
    closed.Day AS ClosedDay
FROM
    dbo.Fact_Tickets t
LEFT JOIN
    dbo.Dim_Dates entry ON t.EntryDateKey = entry.DateKey
LEFT JOIN
    dbo.Dim_Dates response ON t.FirstResponseDateKey = response.DateKey
LEFT JOIN
    dbo.Dim_Dates closed ON t.ClosedDateKey = closed.DateKey
