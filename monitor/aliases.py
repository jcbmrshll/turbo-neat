"""Memorable names for the members of a population: an adjective and a number for the
individual and an animal for its species, e.g. genome 1234 of species 3 is
"sandy-trout-4".

Names are a function of the ids alone, so they're the same wherever they're shown,
and in every run: rerunning a seed names everyone the same. The adjective and number
together pick out the genome, so no two members of a run share a name; the number
counts up by one every len(ADJECTIVES) genomes, so it stays short (4 digits for a
million genomes).
"""

ADJECTIVES = """
agile airy amber ample amused arctic ashen astral atomic autumn azure balmy beefy
big blithe blond blue bold bony bouncy brainy brash brave breezy bright brisk broad
bronze brown bubbly bumpy burly busy calm cheeky cheery chilly chirpy chubby clean
clear clever cloudy clumsy cobalt cocky cold comfy cool copper coral cosmic cozy
crafty crisp crusty cuddly curly cyan dainty damp dandy dapper daring dark deep deft
dewy dim dizzy dotty drowsy dry dusky dusty eager earthy easy elfin exotic faded
fair fancy fast feisty fiery fizzy flaky fleet floral fluffy flying foggy fond fresh
frisky frosty funky fuzzy gentle giant giddy gilded glad glossy golden grand gray
green groovy grumpy gusty hairy hale handy happy hardy hasty hazel hazy hearty heavy
hidden hollow humble hungry icy idle indigo inky iron ivory jade jaunty jazzy jolly
jumbo jumpy keen kind lanky large lavish lazy leafy lean lemon lilac limber little
lively lofty loud lucky lunar lush magic mellow merry mighty mild minty misty moody
mossy muddy murky musty narrow neat nimble noble noisy nutty oaken odd olive orange
ornate pale peppy perky petite pink plucky plump polar posh proud quick quiet quirky
rapid rare ripe rosy rough round royal ruby rusty salty sandy shaggy sharp shiny
short shy silent silky silver sleek sleepy slim sly smoky smooth snappy snowy snug
spiky spry stony stout sunny sweet swift tall tame tangy tawny teal tidy tiny tough
true tubby upbeat urban vast velvet vivid wacky warm wary wavy wild windy wiry wise
witty woolly young zany zesty zippy
""".split()

ANIMALS = """
aardvark albatross alligator alpaca anteater antelope armadillo baboon badger
barracuda bat beagle bear beaver bee beetle bison boar bobcat buffalo bulldog
bumblebee butterfly buzzard camel canary capybara caracal cardinal caribou carp cat
catfish chameleon cheetah chicken chimp chinchilla chipmunk cicada clam cobra
cockatoo cod condor cormorant cougar cow coyote crab crane cricket crocodile crow
cuckoo curlew deer dingo dodo dog dolphin donkey dormouse dove dragonfly duck dugong
eagle eel egret eland elephant elk emu falcon ferret finch firefly flamingo fox frog
gazelle gecko gerbil gibbon giraffe gnu goat goldfinch goose gopher gorilla grouse
gull hamster hare hawk hedgehog heron herring hippo hornet horse hyena ibex ibis
iguana impala jackal jaguar jay jellyfish kangaroo kingfisher kiwi koala koi
kookaburra krill ladybug lark lemming lemur leopard lion lizard llama lobster locust
loon lynx macaw magpie mallard mamba manatee mandrill marmot marten meerkat mink
mole mongoose monkey moose moth mouse mule narwhal newt nightingale ocelot octopus
okapi opossum orca oriole osprey ostrich otter owl ox oyster panda pangolin panther
parrot partridge peacock pelican penguin pheasant pigeon pika piranha platypus pony
porcupine possum puffin puma python quail quokka rabbit raccoon raven reindeer rhino
robin salamander salmon sardine scorpion seahorse seal shark sheep shrew shrimp
skunk sloth snail sparrow squid squirrel starling stingray stork swallow swan tapir
tarsier termite tern tiger toad tortoise toucan trout tuna turkey turtle viper
vulture wallaby walrus warthog wasp weasel whale wolf wombat woodpecker yak zebra
""".split()

# coprime to the number of names they shuffle, so id -> name is one-to-one
_ADJECTIVE_STEP = 157
_ANIMAL_STEP = 73


def species_alias(species_id: int) -> str:
    """An animal, e.g. "dog"; numbered once a run has more species than animals."""
    n = len(ANIMALS)
    lap, k = divmod(species_id, n)
    animal = ANIMALS[(k * _ANIMAL_STEP) % n]
    return f"{animal}{lap + 1}" if lap else animal


def alias(genome_id: int, species_id: int) -> str:
    """The full name, the animal of its species in the middle, e.g. "sandy-trout-4"."""
    n = len(ADJECTIVES)
    number, k = divmod(genome_id, n)
    # spread out, so neighbouring ids get different adjectives
    adjective = ADJECTIVES[(k * _ADJECTIVE_STEP) % n]
    return f"{adjective}-{species_alias(species_id)}-{number}"
