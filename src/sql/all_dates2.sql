SELECT
    CAST(DATETIMEFROMPARTS(d.Year, d.Month, d.Day, 0, 0, 0, 0) AS DATE) AS Date,
    SUM(t.QtTickets) AS QtTickets
FROM
    dbo.Fact_Tickets t
JOIN
    dbo.Dim_Dates d ON t.EntryDateKey = d.DateKey
GROUP BY
    d.Year, d.Month, d.Day
ORDER BY
    d.Year, d.Month, d.Day
