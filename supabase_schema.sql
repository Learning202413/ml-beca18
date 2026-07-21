-- ============================================================
-- Caso 5: PRONABEC Beca 18 - Predicción de Riesgo de Deserción
-- Ejecutar en el SQL Editor de Supabase
-- ============================================================

create table if not exists predicciones_log (
    id                     bigint generated always as identity primary key,
    fecha                  timestamp with time zone not null default now(),
    inputs_usuario         jsonb not null,
    resultado_prediccion   text not null
);

create index if not exists idx_predicciones_log_fecha on predicciones_log (fecha desc);

-- Habilitar Row Level Security
alter table predicciones_log enable row level security;

-- Política: permitir inserción pública (la app usa la anon key para registrar auditoría)
create policy "Permitir insercion publica predicciones_log"
    on predicciones_log for insert
    with check (true);

-- Política: permitir lectura pública (opcional, útil para un dashboard de auditoría)
create policy "Permitir lectura publica predicciones_log"
    on predicciones_log for select
    using (true);
