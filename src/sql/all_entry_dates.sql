-- todas as datas de entrada dos tickets
select
ft.TicketKey,
ft.QtTickets,
d.Year as EntryYear,
d.Month as EntryMonth,
d.Day as EntryDay,
d.Hour as EntryHour,
d.Minute as EntryMinute
from dbo.Fact_Tickets ft
join dbo.Dim_Dates d on ft.EntryDateKey = d.DateKey
