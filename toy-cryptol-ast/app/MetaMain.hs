{-# LANGUAGE OverloadedStrings #-}

module Main where

import qualified Data.Text.IO as T
import qualified Data.Text as Text
import qualified Data.ByteString.Lazy.Char8 as BL8

import System.Environment (getArgs)
import Data.Char (isSpace)
import Data.List (isPrefixOf, isInfixOf, sort, nub)
import Data.Maybe (mapMaybe, catMaybes)
import qualified Data.Set as Set
import qualified Data.Map.Strict as Map

import Data.Aeson (ToJSON(..), (.=), object, encode)

import Cryptol.Parser (parseModule, defaultConfig, Config(..))
import Cryptol.Parser.AST
  ( Module, PName, TopDecl(..), Decl(..), TopLevel(..)
  , Bind(..), BindImpl(..), Schema(..), TySyn(..), Type(..), Prop(..)
  , Expr(..), Match(..), CaseAlt(..), UpdField(..), PropGuardCase(..), Pattern(..)
  , mDecls, tsName, bindImpl
  , value             
  )

import Cryptol.Parser.Position (thing)
import Cryptol.Utils.PP (pretty)
import Cryptol.Utils.RecordMap (recordElements)

--------------------------------------------------------------------------------
-- JSON types
--------------------------------------------------------------------------------

data Ref
  = RefDef String
  | RefImport String
  | RefOp String
  deriving (Eq, Ord)

instance ToJSON Ref where
  toJSON r =
    case r of
      RefDef s    -> object ["def"    .= s]
      RefImport s -> object ["import" .= s]
      RefOp s     -> object ["op"     .= s]

data DefNode = DefNode
  { name       :: String
  , references :: [Ref]
  }

instance ToJSON DefNode where
  toJSON d =
    object
      [ "name"       .= name d
      , "references" .= references d
      ]

data Graph = Graph
  { imports     :: [String]
  , definitions :: [DefNode]
  }

instance ToJSON Graph where
  toJSON g =
    object
      [ "imports"     .= imports g
      , "definitions" .= definitions g
      ]

--------------------------------------------------------------------------------
-- CLI
--   toy-cryptol-graph <file.cry>
--------------------------------------------------------------------------------

main :: IO ()
main = do
  args <- getArgs
  case args of
    [fp] -> run fp
    _    -> putStrLn "Usage: toy-cryptol-graph <file.cry>"

run :: FilePath -> IO ()
run fp = do
  src <- T.readFile fp
  let cfg = defaultConfig { cfgSource = fp }

  case parseModule cfg src of
    Left err -> do
      putStrLn "Parse error:"
      print err
    Right mods -> do
      -- Most of your sliced files should parse into exactly one module;
      -- if not, we just concatenate decls for now.
      let decls = concatMap mDecls mods

          importStrs = importLinesToStrings (Text.unpack src)
          aliasMap   = buildImportAliasMap importStrs

          localNames = Set.fromList (map pretty (localDefNames decls))

          defNodes =
            [ DefNode
                { name = pretty defName
                , references =
                    sort . nub $
                      classifyRefs aliasMap localNames (refsOfDef defKind decls defName)
                }
            | (defName, defKind) <- topLevelDefs decls
            ]

          g = Graph { imports = importStrs, definitions = defNodes }

      BL8.putStrLn (encode g)

--------------------------------------------------------------------------------
-- Imports: keep only "Common::GF28 as GF28"
--------------------------------------------------------------------------------

-- Detect whether a line is "import ..."
isImportLine :: String -> Bool
isImportLine line =
  case dropWhile isSpace line of
    ('i':'m':'p':'o':'r':'t':' ':_) -> True
    _                               -> False

-- Convert "import Common::GF28 as GF28" -> "Common::GF28 as GF28"
importLinesToStrings :: String -> [String]
importLinesToStrings srcStr =
  [ dropWhile isSpace (drop (length ("import " :: String)) (dropWhile isSpace line))
  | line <- lines srcStr
  , isImportLine line
  ]

-- Build alias -> full import-string map.
-- Example: "Common::GF28 as GF28" => alias "GF28"
buildImportAliasMap :: [String] -> Map.Map String String
buildImportAliasMap imps =
  Map.fromList (catMaybes (map parseAlias imps))
  where
    parseAlias :: String -> Maybe (String, String)
    parseAlias s =
      case splitOnAs s of
        Just (modPart, alias) ->
          let alias' = trim alias
              full   = trim modPart ++ " as " ++ alias'
          in if null alias' then Nothing else Just (alias', full)
        Nothing -> Nothing

    splitOnAs :: String -> Maybe (String, String)
    splitOnAs s =
      case breakOn " as " s of
        Nothing -> Nothing
        Just (a,b) -> Just (a,b)

    breakOn :: String -> String -> Maybe (String, String)
    breakOn pat s =
      case findIndexSubstring pat s of
        Nothing -> Nothing
        Just i  ->
          let (l, r0) = splitAt i s
              r       = drop (length pat) r0
          in Just (l, r)

    findIndexSubstring :: String -> String -> Maybe Int
    findIndexSubstring pat s = go 0 s
      where
        go _ [] = Nothing
        go i xs
          | pat `isPrefixOf` xs = Just i
          | otherwise           = go (i+1) (tail xs)

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

dropWhileEnd :: (a -> Bool) -> [a] -> [a]
dropWhileEnd p = reverse . dropWhile p . reverse

--------------------------------------------------------------------------------
-- Top-level defs and reference extraction
--------------------------------------------------------------------------------

-- We want "the definitions are": types + values + properties,
-- but NOT bare signatures as separate definitions.
data DefKind = KType | KValue deriving (Eq)

-- List all top-level definitions (name + kind).
-- For DRec we return each binding separately.
topLevelDefs :: [TopDecl PName] -> [(PName, DefKind)]
topLevelDefs = concatMap goTD
  where
    goTD td =
      case td of
        Decl tl -> goDecl (dropLocDecl (tlValue tl))
        _       -> []

    goDecl d =
      case d of
        DType ts         -> [(thing (tsName ts), KType)]
        DBind b          -> [(thing (bName b), KValue)]
        DRec bs          -> [ (thing (bName b), KValue) | b <- bs ]
        DPatBind p _     ->
          case p of
            PVar x          -> [(thing x, KValue)]
            PLocated p' _   -> goDecl (DPatBind p' (ELit undefined))
            _               -> []
        DSignature{}     -> []   -- ignore: signature isn't a "definition node"
        DLocated d' _    -> goDecl d'
        _                -> []

-- All local definition names (same “nodes” as above, plus type names).
localDefNames :: [TopDecl PName] -> [PName]
localDefNames = map fst . topLevelDefs

-- Find references for a given top-level definition.
-- For types: names in the type RHS.
-- For values: names used in the defining expression and its type schema.
refsOfDef :: DefKind -> [TopDecl PName] -> PName -> Set.Set String
refsOfDef kind decls target =
  case kind of
    KType  -> Set.unions (map (refsType target) decls)
    KValue -> Set.unions (map (refsValue target) decls)

refsType :: PName -> TopDecl PName -> Set.Set String
refsType target td =
  case td of
    Decl tl ->
      case dropLocDecl (tlValue tl) of
        DType (TySyn nm _ _ ty)
          | thing nm == target ->
              Set.map prettyName (namesInType ty)
        DLocated d' _ -> refsType target (Decl tl { tlValue = d' })
        _ -> Set.empty
    _ -> Set.empty

refsValue :: PName -> TopDecl PName -> Set.Set String
refsValue target td =
  case td of
    Decl tl ->
      case dropLocDecl (tlValue tl) of
        DBind b
          | thing (bName b) == target ->
              case bindImpl b of
                Just (DExpr e)        -> Set.map prettyName (namesInExpr e)
                Just (DPropGuards cs) -> Set.map prettyName (Set.unions [ namesInExpr (pgcExpr c) | c <- cs ])
                _                     -> Set.empty
          | otherwise -> Set.empty

        DRec bs ->
          Set.unions
            [ if thing (bName b) == target
                then case bindImpl b of
                       Just (DExpr e)        -> Set.map prettyName (namesInExpr e)
                       Just (DPropGuards cs) -> Set.map prettyName (Set.unions [ namesInExpr (pgcExpr c) | c <- cs ])
                       _                     -> Set.empty
                else Set.empty
            | b <- bs
            ]

        DPatBind (PVar x) e
          | thing x == target -> Set.map prettyName (namesInExpr e)
          | otherwise         -> Set.empty

        DSignature ns sch
          | target `elem` map thing ns ->
              Set.map prettyName (namesInSchema sch)

        DLocated d' _ -> refsValue target (Decl tl { tlValue = d' })
        _ -> Set.empty
    _ -> Set.empty

prettyName :: PName -> String
prettyName = pretty

-- Convert raw name refs into either local def refs or import refs.
-- We drop external refs (builtins) unless you later decide you want them.
classifyRefs
  :: Map.Map String String  -- alias -> "Common::GF28 as GF28"
  -> Set.Set String         -- local def names (as strings)
  -> Set.Set String         -- raw referenced names (as strings)
  -> [Ref]
classifyRefs aliasMap localDefs refs =
  [ if r `Set.member` localDefs
      then RefDef r
      else case Map.lookup r aliasMap of
             Just imp -> RefImport imp
             Nothing  -> RefOp r
  | r <- Set.toList refs
  ]

--------------------------------------------------------------------------------
-- Referenced names inside schemas/types/exprs
-- (You already had these in your slicer; keep them here too.)
--------------------------------------------------------------------------------

namesInSchema :: Schema PName -> Set.Set PName
namesInSchema (Forall _ props ty _) =
  namesInType ty `Set.union` Set.unions (map namesInProp props)

namesInProp :: Prop PName -> Set.Set PName
namesInProp (CType ty) = namesInType ty

namesInType :: Type PName -> Set.Set PName
namesInType ty =
  case ty of
    TFun a b       -> namesInType a `Set.union` namesInType b
    TSeq a b       -> namesInType a `Set.union` namesInType b
    TBit           -> Set.empty
    TNum _         -> Set.empty
    TChar _        -> Set.empty
    TUser n args   -> Set.insert (thing n) (Set.unions (map namesInType args))
    TTyApp ts      -> Set.unions [ namesInType (value t) | t <- ts ]
    TRecord _      -> Set.empty
    TTuple ts      -> Set.unions (map namesInType ts)
    TWild          -> Set.empty
    TLocated t _   -> namesInType t
    TParens t _    -> namesInType t
    TInfix a _ _ b -> namesInType a `Set.union` namesInType b

namesInExpr :: Expr PName -> Set.Set PName
namesInExpr expr =
  case expr of
    EVar n        -> Set.singleton n
    ELit _        -> Set.empty
    EGenerate e   -> go e
    ETuple es     -> Set.unions (map go es)
    ERecord rec   -> Set.unions [ go e | (_, e) <- recordElements rec ]
    ESel e _      -> go e
    EUpd me fields ->
      maybe Set.empty go me `Set.union`
      Set.unions [ go e | UpdField _ _ e <- fields ]
    EList es      -> Set.unions (map go es)
    EFromTo{}         -> Set.empty
    EFromToBy{}       -> Set.empty
    EFromToDownBy{}   -> Set.empty
    EFromToLessThan{} -> Set.empty
    EInfFrom e me ->
      go e `Set.union` maybe Set.empty go me
    EComp e mss ->
      go e `Set.union` Set.unions [ Set.unions (map namesInMatch ms) | ms <- mss ]
    EApp e1 e2   -> go e1 `Set.union` go e2
    EAppT e _    -> go e
    EIf e1 e2 e3 -> go e1 `Set.union` go e2 `Set.union` go e3
    ECase e alts ->
      go e `Set.union` Set.unions [ go rhs | CaseAlt _ rhs <- alts ]
    EWhere e ds  ->
      go e `Set.union` Set.unions (map namesInDecl ds)
    ETyped e _   -> go e
    ETypeVal _   -> Set.empty
    EFun _ _ e   -> go e
    ELocated e _ -> go e
    ESplit e     -> go e
    EParens e    -> go e
    EInfix e1 op _ e2 ->
      go e1 `Set.union` go e2 `Set.union` Set.singleton (thing op)
    EPrefix _ e  -> go e
  where
    go = namesInExpr

namesInMatch :: Match PName -> Set.Set PName
namesInMatch m =
  case m of
    Match _ e  -> namesInExpr e
    MatchLet b -> namesInDecl (DBind b)

namesInDecl :: Decl PName -> Set.Set PName
namesInDecl d =
  case d of
    DBind b ->
      case bindImpl b of
        Just (DExpr e)        -> namesInExpr e
        Just (DPropGuards cs) -> Set.unions [ namesInExpr (pgcExpr c) | c <- cs ]
        Nothing               -> Set.empty
    DRec bs ->
      Set.unions [ namesInDecl (DBind b) | b <- bs ]
    DPatBind _ e ->
      namesInExpr e
    DSignature _ s ->
      namesInSchema s
    DType (TySyn _ _ _ ty) ->
      namesInType ty
    DLocated d' _ ->
      namesInDecl d'
    _ ->
      Set.empty

dropLocDecl :: Decl PName -> Decl PName
dropLocDecl (DLocated d _) = dropLocDecl d
dropLocDecl d              = d
