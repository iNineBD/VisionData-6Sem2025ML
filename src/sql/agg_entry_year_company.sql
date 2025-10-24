SELECT
    t.CompanyKey,
    d.Year,
    SUM(t.QtTickets) AS TotalTickets
FROM dbo.Fact_Tickets t
JOIN dbo.Dim_Dates d 
    ON t.EntryDateKey = d.DateKey
GROUP BY
    t.CompanyKey,
    d.Year
ORDER BY
    t.CompanyKey,
    d.Year;