# Tory serwowania na localhost/lil-krzych

Jeden skrypt na model, stałe opcje, jedyny argument to port (tak woła llama-swap: `${PORT}`).
Wspólne rzeczy są w `lib.sh`: obraz wydania, czekanie do 90 s na zwolnienie kart, pierwszy plan
(kontener umiera razem z procesem przy SIGTERM), log w `/home/krzych/glm-5.2/lane-logs/`.

| Skrypt | Model | Karty | Domyślny port |
|---|---|---|---|
| `glm53-full.sh` | GLM-5.3 full NVFP4-CSF, TP6/DCP3, MTP3, ctx 262 144, KV 15 GiB | 0,1,2,3,4,7 | 10004 |
| `glm53-flash.sh` | GLM-5.3 Flash CSF-QAD qad2, TP4, MTP3, ctx 1M | 0-3 | 11110 |
| `glm53-flash-tp2.sh` | GLM-5.3 Flash qad2, preset glm53-tp2 (TP2/DCP2) | 4,7 | 11112 |
| `ds41-flash.sh` | DeepSeek-V4.1-Flash-1017, oryginalny checkpoint, TP4 | 0-3 | 11110 |
| `ds41-flash-abl.sh` | DeepSeek-V4.1-Flash-1017 abliterated (CSF), TP4, koder 1017 | 0-3 | 11110 |

Zatrzymanie: `stop.sh <nazwa>` (nazwa = nazwa skryptu bez `.sh`), czeka na zwolnienie pamięci kart.

Zmienne z zewnątrz (tylko dwie):
- `IMAGE=...` inny obraz, np. poprzednie wydanie do porównania;
- `DRY_RUN=1` wypisuje pełne `docker run` i nic nie uruchamia.

Determinizm (łatki z toru full) jest w obrazie domyślnie włączony; Flash i DS4.1 wyłączają go
przełącznikami w `ENV`. Planista izolujący nie jest używany (dawny `ISOLATION=1` to
`--scheduler-cls=vllm.v1.core.sched.lil_isolated_scheduler.IsolatedAsyncScheduler` w `ARGS`).
Wolumeny `/cache` są te same co w dawnych torach (karmic-53-*-runtime, karmic-ds41f-runtime), więc
skompilowane jądra i strojenie B12X zostają.

Bramki po rozruchu (skrypty w ~/glm-5.2):
`v20_53_lil/check-concurrency-coherence-53.py --port P [--model NAZWA]`,
`karmic_53_lil/check-glm53-kk.sh glm53-full`, `karmic_53_lil/check-glm53-flash-kk.sh glm53-flash`,
`karmic_ds41f/check-ds41f-kk.sh ds41-flash`.
