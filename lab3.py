import numpy as np
import matplotlib.pyplot as plt

alpha, u_alpha = 0.910, 0.018
delta, u_delta = 5.180, 0.045
rho_ad = -0.65

# k e s_x scelti
k, u_k = 1.000, 0.010
s_x, u_sx = 1.000, 0.005

z = np.array([alpha, delta, k, s_x])
C_ad = np.array([
    [u_alpha ** 2,                  rho_ad * u_alpha * u_delta],
    [rho_ad * u_alpha * u_delta,    u_delta ** 2],
])

C_z = np.zeros((4, 4))
C_z[:2, :2] = C_ad
C_z[2, 2] = u_k ** 2
C_z[3, 3] = u_sx ** 2

print("z = ", z)
print("\nC_z =")
print(C_z)

#verifiche
tol = 1e-9
simmetrica = np.allclose(C_z, C_z.T)
diag_positiva = np.all(np.diag(C_z) > 0)
autovalori = np.linalg.eigvalsh(C_z)
autovalori_ok = np.all(autovalori >= -tol * np.max(np.abs(autovalori))) #autoval non negativi

print("\nVerifiche su C_z:")
print("Simmetrica: ", simmetrica)
print("Diagonale positiva: ", diag_positiva)
print("Autovalori: ", autovalori)
print("Autovalori non negativi: ", autovalori_ok)

#matrice di correlazione
u_z = np.sqrt(np.diag(C_z))
corr = C_z / np.outer(u_z, u_z)

print("\nMatrice di correlazione:")
print(np.round(corr, 3))
print(f"\nrho(alpha, delta) = {corr[0, 1]:.3f} con atteso é {rho_ad}")

nomi = [r"$\alpha$", r"$\delta$", r"$k$", r"$s_x$"]
plt.figure(figsize=(5.5, 4.5))
im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, label=r"$\rho_{ij}$")
plt.xticks(range(4), nomi)
plt.yticks(range(4), nomi)
for i in range(4):
    for j in range(4):
        plt.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center")
plt.title("Matrice di correlazione dei parametri di z")
plt.tight_layout()
plt.show()




from scipy.stats import skew
 
M = 100_000 #campioni gaussiani correlati generati
media_input = z
 
def derivate(campioni):
    al, de, kk, ss = campioni.T
    a = al / (kk * ss)
    d = de / (kk * ss)
    q = d / a
    return a, d, q
 
 
def jacobiano(zz):
    al, de, kk, ss = zz
    return np.array([
        [1 / (kk * ss), 0, -al / (kk ** 2 * ss), -al / (kk * ss ** 2)],
        [0, 1 / (kk * ss), -de / (kk ** 2 * ss), -de / (kk * ss ** 2)],
        [-de / al ** 2, 1 / al, 0, 0],
    ])
 
 
def propaga_lineare(zz, C):
    J = jacobiano(zz)
    return np.sqrt(np.diag(J @ C @ J.T))
 
 
def monte_carlo(C, seed=20261003):
    rng = np.random.default_rng(seed)
    campioni = rng.multivariate_normal(mean=media_input, cov=C, size=M)
    return campioni
 
 
def tabella(titolo, campioni, C):
    ammissibili = np.all(campioni > 0)
    a, d, q = derivate(campioni)
    nom = [derivate(z[None, :])[i][0] for i in range(3)]
    u_lin = propaga_lineare(z, C)
 
    print(f"\n{titolo}")
    print(f"Input fisicamente ammissibili (tutti > 0): {ammissibili} "
          f"({np.sum(~np.all(campioni > 0, axis=1))} campioni non validi)")
    print(f"{'':<4}{'nominale':>10}{'u lineare':>11}{'media MC':>10}{'dev. MC':>9}"
          f"{'mediana':>9}{'p16':>9}{'p84':>9}")
    risultati = {}
    for i, (nome, v) in enumerate(zip(["a", "d", "q"], [a, d, q])):
        p16, med, p84 = np.percentile(v, [16, 50, 84])
        risultati[nome] = dict(media=v.mean(), std=v.std(ddof=1), med=med,
                               p16=p16, p84=p84, u_lin=u_lin[i], nom=nom[i], v=v)
        print(f"{nome:<4}{nom[i]:>10.4f}{u_lin[i]:>11.4f}{v.mean():>10.4f}"
              f"{v.std(ddof=1):>9.4f}{med:>9.4f}{p16:>9.4f}{p84:>9.4f}")
    return risultati
 
 
campioni = monte_carlo(C_z)
ris = tabella("Monte Carlo correlato (rho_alpha_delta = -0.65)", campioni, C_z)
 
q = ris["q"]["v"]
print("\nConfronto lineare / Monte Carlo per q:")
print(f"centro:        nominale {ris['q']['nom']:.4f}, media MC {ris['q']['media']:.4f}, "
      f"mediana MC {ris['q']['med']:.4f}")
print(f"dispersione:   u lineare {ris['q']['u_lin']:.4f}, std MC {ris['q']['std']:.4f}, "
      f"(p84-p16)/2 = {(ris['q']['p84'] - ris['q']['p16']) / 2:.4f}")
print(f"simmetria:     skewness = {skew(q):.3f}, "
      f"p84-mediana = {ris['q']['p84'] - ris['q']['med']:.4f}, "
      f"mediana-p16 = {ris['q']['med'] - ris['q']['p16']:.4f}")
 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
 
ax1.hist(q, bins=80, color="tab:blue", alpha=0.7, edgecolor="black", linewidth=0.3)
ax1.axvline(ris["q"]["nom"], color="black", linestyle="-", label=f"nominale = {ris['q']['nom']:.3f}")
ax1.axvline(ris["q"]["media"], color="tab:orange", linestyle="--", label=f"media = {ris['q']['media']:.3f}")
ax1.axvline(ris["q"]["med"], color="tab:red", linestyle=":", linewidth=2, label=f"mediana = {ris['q']['med']:.3f}")
ax1.set_xlabel(r"$q = d/a$")
ax1.set_ylabel("frequenza")
ax1.set_title("Distribuzione Monte Carlo di q")
ax1.legend()
 
ax2.plot(campioni[:3000, 0], campioni[:3000, 1], ".", markersize=2, alpha=0.5)
ax2.plot(alpha, delta, "r+", markersize=12, label="valore nominale")
ax2.set_xlabel(r"$\alpha$ [mm$^{-1}$]")
ax2.set_ylabel(r"$\delta$ [mm$^{-1}$]")
ax2.set_title(r"Campioni nel piano $(\alpha, \delta)$")
ax2.legend()
 
plt.tight_layout()
plt.show()

C_nocorr = C_z.copy()
C_nocorr[0, 1] = C_nocorr[1, 0] = 0.0
 
campioni_nc = monte_carlo(C_nocorr)
ris_nc = tabella("Monte Carlo senza correlazione (rho_alpha_delta = 0)", campioni_nc, C_nocorr)
 
print("\nConfronto delle incertezze (correlato vs senza correlazione):")
print(f"{'':<4}{'u lin corr':>11}{'u lin nocorr':>14}{'std MC corr':>13}{'std MC nocorr':>15}{'diff. rel. MC':>15}")
for nome in ["a", "d", "q"]:
    c, n = ris[nome], ris_nc[nome]
    diff = (n["std"] - c["std"]) / c["std"]
    print(f"{nome:<4}{c['u_lin']:>11.4f}{n['u_lin']:>14.4f}{c['std']:>13.4f}{n['std']:>15.4f}{diff:>15.1%}")
 
print("\nIncertezze marginali di alpha e delta (devono restare uguali):")
print(f"u(alpha) correlato / non correlato: {campioni[:, 0].std(ddof=1):.4f} / {campioni_nc[:, 0].std(ddof=1):.4f}")
print(f"u(delta) correlato / non correlato: {campioni[:, 1].std(ddof=1):.4f} / {campioni_nc[:, 1].std(ddof=1):.4f}")

gruppi = {
    r"fit $(\alpha, \delta)$": [0, 1],
    r"$k$": [2],
    r"$s_x$": [3],
}
grandezze = ["a", "d", "q"]
 
J = jacobiano(z)
varianze = {}
for nome_g, idx in gruppi.items():
    C_g = np.zeros((4, 4))
    C_g[np.ix_(idx, idx)] = C_z[np.ix_(idx, idx)]
    varianze[nome_g] = np.diag(J @ C_g @ J.T)
 
var_tot = np.diag(J @ C_z @ J.T)
F = {g: v / var_tot for g, v in varianze.items()}
 
print("\nContributi alla varianza (F_j, %):")
print(f"{'gruppo':<24}" + "".join(f"{g:>9}" for g in grandezze))
for nome_g in gruppi:
    etichetta = nome_g.replace("$", "").replace("\\", "")
    print(f"{etichetta:<24}" + "".join(f"{100 * f:>9.1f}" for f in F[nome_g]))
print(f"{'somma':<24}" + "".join(f"{100 * sum(F[g][i] for g in gruppi):>9.1f}" for i in range(3)))
 
fig, ax = plt.subplots(figsize=(8, 4.5))
larghezza = 0.25
for j, nome_g in enumerate(gruppi):
    ax.bar(np.arange(3) + (j - 1) * larghezza, 100 * F[nome_g], larghezza, label=nome_g)
ax.set_xticks(range(3))
ax.set_xticklabels(grandezze)
ax.set_ylabel(r"$F_j$ [%]")
ax.set_xlabel("grandezza derivata")
ax.set_title("Frazione di varianza per gruppo di input")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("mappa_contributi.png", dpi=150)
plt.show()



def covarianza_scenario(u_alpha_s=u_alpha, u_delta_s=u_delta, rho=rho_ad,
                        u_k_s=u_k, u_sx_s=u_sx):
    C = np.zeros((4, 4))
    C[0, 0] = u_alpha_s ** 2
    C[1, 1] = u_delta_s ** 2
    C[0, 1] = C[1, 0] = rho * u_alpha_s * u_delta_s
    C[2, 2] = u_k_s ** 2
    C[3, 3] = u_sx_s ** 2
    return C
 
 
scenari = {
    "Nominale": covarianza_scenario(),
    "Correlazione ignorata": covarianza_scenario(rho=0.0),
    "Calibrazione k migliorata": covarianza_scenario(u_k_s=u_k / 2),
    "Calibrazione s_x migliorata": covarianza_scenario(u_sx_s=u_sx / 2),
    "Statistica fit migliorata": covarianza_scenario(u_alpha_s=u_alpha / 2, u_delta_s=u_delta / 2),
}
 
risultati_scenari = {}
print("\nIncertezze nei vari scenari (lineare / Monte Carlo):")
print(f"{'scenario':<30}{'u(a)':>16}{'u(d)':>16}{'u(q)':>16}")
for nome_s, C_s in scenari.items():
    u_lin_s = propaga_lineare(z, C_s)
    a_s, d_s, q_s = derivate(monte_carlo(C_s))
    u_mc_s = np.array([a_s.std(ddof=1), d_s.std(ddof=1), q_s.std(ddof=1)])
    risultati_scenari[nome_s] = u_lin_s
    print(f"{nome_s:<30}" + "".join(f"{u_lin_s[i]:>9.4f}/{u_mc_s[i]:.4f}" for i in range(3)))
 
nominale_u = risultati_scenari["Nominale"]
print("\nVariazione relativa rispetto al nominale (lineare):")
print(f"{'scenario':<30}{'u(a)':>9}{'u(d)':>9}{'u(q)':>9}")
for nome_s, u_s in risultati_scenari.items():
    if nome_s == "Nominale":
        continue
    var = (u_s - nominale_u) / nominale_u
    print(f"{nome_s:<30}" + "".join(f"{100 * v:>8.1f}%" for v in var))
 
migliorati = [s for s in risultati_scenari if s not in ("Nominale", "Correlazione ignorata")]
print("\nIntervento più efficace (minima incertezza):")
for i, g in enumerate(grandezze):
    migliore = min(migliorati, key=lambda s: risultati_scenari[s][i])
    print(f"{g}: {migliore} (u = {risultati_scenari[migliore][i]:.4f}, "
          f"nominale {nominale_u[i]:.4f})")
 
fig, ax = plt.subplots(figsize=(9, 4.5))
nomi_s = list(risultati_scenari)
larghezza = 0.25
for i, g in enumerate(grandezze):
    rel = [risultati_scenari[s][i] / nominale_u[i] for s in nomi_s]
    ax.bar(np.arange(len(nomi_s)) + (i - 1) * larghezza, rel, larghezza, label=f"u({g})")
ax.axhline(1, color="black", linewidth=1)
ax.set_xticks(range(len(nomi_s)))
ax.set_xticklabels(nomi_s, rotation=20, ha="right")
ax.set_ylabel("incertezza / nominale")
ax.set_title("Stabilità delle incertezze nei vari scenari")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
