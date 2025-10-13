-- todas as datas de primeiro contato dos tickets
select
ft.TicketKey,
ft.QtTickets,
d.Year as FirstResponseYear,
d.Month as FirstResponseMonth,
d.Day as FirstResponseDay,
d.Hour as FirstResponseHour,
d.Minute as FirstResponseMinute
from dbo.Fact_Tickets ft
join dbo.Dim_Dates d on ft.FirstResponseDateKey = d.DateKey
